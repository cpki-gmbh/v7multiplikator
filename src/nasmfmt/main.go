// nasmfmt -- formatter for NASM source files (v7 patched fork)
//
// Upstream: github.com/yamnikov-oleg/nasmfmt (MIT license, see LICENSE)
// Patched version: v0.0.0-20230907082239-e010ffea9224
//
// v7 patches -- all three fix CORRUPTION bugs that fire on this codebase:
//
//  1. parseLabel: a `label: db/dw/dd/...` line sliced the label INCLUDING its
//     colon and print() appended a second one, producing `label::` (invalid
//     NASM; 413 occurrences in the v7 tree). Fixed by trimming the trailing
//     colon before print() re-adds it.
//
//  2. Text normalization (whitespace collapse + comma spacing) is now
//     QUOTE-AWARE: string literals keep their bytes verbatim. The stock tool
//     collapsed "a  b" -> "a b" and rewrote "a,b" -> "a, b" inside quoted
//     strings (77 occurrences in the v7 tree).
//
//  3. NASM backquote (`) strings are treated as strings everywhere (comment
//     detection, label detection, normalization). Only backquoted strings
//     process backslash escapes, matching NASM: a backslash inside '...' or
//     "..." is a literal byte and does not hide a following quote.
package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"unicode/utf8"
)

var (
	insIndent     int
	commentIndent int
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [params] [file1 [file2 ...]]\nParameters:\n", os.Args[0])
		flag.PrintDefaults()
	}
	flag.IntVar(&insIndent, "ii", 8, "Indentation for instructions in spaces")
	flag.IntVar(&commentIndent, "ci", 40, "Indentation for comments in spaces")
}

type asmLine struct {
	label   string
	text    string
	comment string
}

func indexRune(s []rune, rn rune) int {
	for i := range s {
		if s[i] == rn {
			return i
		}
	}
	return -1
}

func indexRuneAny(s []rune, any string) int {
	anyr := []rune(any)
	for i, rn := range s {
		if indexRune(anyr, rn) >= 0 {
			return i
		}
	}
	return -1
}

func runesRepeat(rep []rune, count int) []rune {
	ret := []rune{}
	for i := 0; i < count; i++ {
		ret = append(ret, rep...)
	}
	return ret
}

// indexQuoteClose returns the index (within sr, which starts AT the opening
// quote) of the closing quote. NASM escape rules: only backquoted strings
// process backslash escapes, so `\` inside ' or " strings is a literal byte
// and does not hide a following quote char.
func indexQuoteClose(sr []rune, quote rune) int {
	for i := 1; i < len(sr); i++ {
		if quote == '`' && sr[i] == '\\' && i+1 < len(sr) {
			i++
			continue
		}
		if sr[i] == quote {
			return i
		}
	}
	return -1
}

// noquotes replaces all quoted parts of a string with provided replacement.
// E.g. noquotes(`I'm "in love" with donuts`, "x") -> `I'm xxxxxxxxx with donuts`.
//
// It's useful for performing substring searches ignoring quotations.
// Index of a substring in a 'noquotes' version with len(rep)==1
// would equal its index in original string.
func noquotes(s, rep string) string {
	sr := []rune(s)
	repr := []rune(rep)
	outr := []rune{}

	for len(sr) > 0 {
		// Find first quotation mark
		ind := indexRuneAny(sr, "\"'`")
		if ind >= 0 {
			quote := sr[ind]
			// Find its pair
			end := indexQuoteClose(sr[ind:], quote)
			// If it has no pair - include it into output string and skip
			if end < 0 {
				outr = append(outr, sr[:ind+1]...)
				sr = sr[ind+1:]
				continue
			}
			// If it's paired - replace it with reps
			ind2 := ind + end
			outr = append(outr, sr[:ind]...)
			outr = append(outr, runesRepeat(repr, ind2-ind+1)...)
			sr = sr[ind2+1:]
		} else {
			// If no quotation marks found -
			// include the rest of input string and break the cycle.
			outr = append(outr, sr...)
			sr = nil
		}
	}

	return string(outr)
}

var dataPseudos = compilePseudoes()

func compilePseudoes() (ret []*regexp.Regexp) {
	var (
		raw = []string{"db", "dw", "dd", "dq", "ddq", "do", "dt"}
		re  = `(?:\s|^)%v(?:\s|")`
	)
	for _, p := range raw {
		ret = append(ret, regexp.MustCompile(fmt.Sprintf(re, p)))
	}
	return
}

func parseLabel(line string) (lbl string, rest string) {
	lbl = ""
	rest = line
	noq := noquotes(line, "x")

	ind := strings.Index(noq, ":")
	if ind >= 0 {
		lbl = strings.TrimSpace(line[:ind])
		rest = line[ind+1:]
	}

	for _, pseudo := range dataPseudos {
		inds := pseudo.FindStringIndex(noq)
		if len(inds) > 0 {
			ind := inds[0]
			lbl = strings.TrimSpace(line[:ind])
			// v7 PATCH: when the `:`-split above already produced the label,
			// slicing from the full line keeps the trailing colon, and
			// print() appends a second one -> `label::` (invalid NASM).
			lbl = strings.TrimSuffix(lbl, ":")
			rest = line[ind:]
		}
	}

	// Labels should not contain space
	matched := regexp.MustCompile(`\S*\s+\S*`).MatchString(lbl)
	if matched {
		lbl, rest = "", line
	}

	return
}

func parseComment(line string) (cmt string, rest string) {
	noq := noquotes(line, "x")

	ind := strings.Index(noq, ";")
	if ind >= 0 {
		cmt = strings.TrimSpace(line[ind+1:])
		rest = line[:ind]
		return
	}
	return "", line
}

func parseLine(line string) *asmLine {
	l := &asmLine{}

	l.comment, line = parseComment(line)
	l.label, line = parseLabel(line)

	l.text = strings.TrimSpace(line)
	l.text = normOutsideQuotes(l.text)

	return l
}

// normOutsideQuotes collapses whitespace runs to a single space and rewrites
// `,\s*` as `, ` -- but only OUTSIDE string literals. The upstream
// mulSpaces/commaSpaces regexes ran over the whole text and silently corrupted
// string bytes: "a  b" -> "a b", "a,b" -> "a, b".
func normOutsideQuotes(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	sr := []rune(s)
	i := 0
	for i < len(sr) {
		c := sr[i]
		if c == '"' || c == '\'' || c == '`' {
			end := indexQuoteClose(sr[i:], c)
			if end < 0 {
				// Unpaired quote (invalid NASM anyway): emit rest verbatim.
				b.WriteString(string(sr[i:]))
				break
			}
			b.WriteString(string(sr[i : i+end+1]))
			i += end + 1
			continue
		}
		if c == ' ' || c == '\t' {
			j := i
			for j < len(sr) && (sr[j] == ' ' || sr[j] == '\t') {
				j++
			}
			b.WriteRune(' ')
			i = j
			continue
		}
		if c == ',' {
			j := i + 1
			for j < len(sr) && (sr[j] == ' ' || sr[j] == '\t') {
				j++
			}
			b.WriteString(", ")
			i = j
			continue
		}
		b.WriteRune(c)
		i++
	}
	return b.String()
}

func (l *asmLine) empty() bool {
	return l.label == "" && l.text == "" && l.comment == ""
}

func (l *asmLine) print(w io.Writer) {
	var (
		space  = []byte{' '}
		newl   = []byte{'\n'}
		column = 0
	)

	if l.label != "" {
		w.Write([]byte(l.label))
		w.Write([]byte{':'})
		column += utf8.RuneCountInString(l.label) + 1
		if l.text != "" {
			w.Write(newl)
			column = 0
		}
	}

	if l.text != "" {
		if strings.HasPrefix(l.text, "%") {
			// v7 PATCH: preprocessor directives (%define, %macro, %endif, ...)
			// print at column 0 -- they are not instructions and carry no
			// instruction indent.
			w.Write([]byte(l.text))
			column += utf8.RuneCountInString(l.text)
		} else {
			w.Write(bytes.Repeat(space, insIndent))
			w.Write([]byte(l.text))
			column += insIndent
			column += utf8.RuneCountInString(l.text)
		}
	}

	if l.comment != "" {
		if column != 0 {
			if column < commentIndent-1 {
				w.Write(bytes.Repeat(space, commentIndent-column-1))
			} else {
				w.Write(space)
			}
		}
		w.Write([]byte{';', ' '})
		w.Write([]byte(l.comment))
	}

	w.Write(newl)
}

func formatto(filename, outname string) error {
	var file *os.File
	var out *os.File

	if filename == "-" {
		file = os.Stdin
		out = os.Stdout
	} else {
		var err error

		file, err = os.Open(filename)
		if err != nil {
			return err
		}
		defer file.Close()

		out, err = os.Create(outname)
		if err != nil {
			return err
		}
		defer out.Close()
	}

	scanner := bufio.NewScanner(file)
	lastEmpty := false
	first := true
	for scanner.Scan() {
		line := scanner.Text()
		asmLine := parseLine(line)
		if lastEmpty && asmLine.empty() {
			continue // Output no more than one empty line
		}
		// v7 PATCH: every %macro definition gets a blank line before it.
		if !first && !lastEmpty && strings.HasPrefix(asmLine.text, "%macro") {
			fmt.Fprintln(out)
		}
		asmLine.print(out)
		lastEmpty = asmLine.empty()
		first = false
	}
	return nil
}

func format(filename string) error {
	outname := filename + "~"
	if err := formatto(filename, outname); err != nil {
		return err
	}
	if filename != "-" {
		if err := os.Rename(outname, filename); err != nil {
			return err
		}
	}
	return nil
}

func main() {
	flag.Parse()
	files := flag.Args()
	for _, fn := range files {
		err := format(fn)
		if err != nil {
			fmt.Println(err)
		}
	}
}
