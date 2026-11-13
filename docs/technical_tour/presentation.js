'use strict';

const slides = [...document.querySelectorAll('.slide')];
const prev = document.querySelector('#prev');
const next = document.querySelector('#next');
const picker = document.querySelector('#picker');
const counter = document.querySelector('#counter');
const mode = document.querySelector('#mode');

function fromHash() {
  const match = /^#slide-(\d+)$/.exec(location.hash);
  const number = match ? Number(match[1]) : 1;
  return number >= 1 && number <= slides.length ? number - 1 : 0;
}

let current = fromHash();
let reading = matchMedia('(max-width: 620px)').matches;
document.body.classList.add('js');

function updateMode() {
  document.body.classList.toggle('reading', reading);
  mode.textContent = reading ? 'Slide view' : 'Reading view';
}

function show(index, updateHash = true) {
  current = Math.max(0, Math.min(index, slides.length - 1));
  slides.forEach((slide, i) => { slide.hidden = i !== current; });
  picker.value = String(current);
  counter.textContent = `${current + 1} / ${slides.length}`;
  prev.disabled = current === 0;
  next.disabled = current === slides.length - 1;
  if (updateHash) {
    history.replaceState(null, '', `#slide-${current + 1}`);
    window.scrollTo(0, 0);
  }
}

prev.addEventListener('click', () => show(current - 1));
next.addEventListener('click', () => show(current + 1));
picker.addEventListener('change', () => show(Number(picker.value)));
mode.addEventListener('click', () => { reading = !reading; updateMode(); });
window.addEventListener('hashchange', () => show(fromHash(), false));

document.addEventListener('keydown', event => {
  if (event.altKey || event.ctrlKey || event.metaKey ||
      event.target.closest?.('select, input, textarea, a, summary')) return;
  const targets = {
    ArrowLeft: current - 1,
    PageUp: current - 1,
    ArrowRight: current + 1,
    PageDown: current + 1,
    Home: 0,
    End: slides.length - 1,
  };
  if (Object.hasOwn(targets, event.key)) {
    event.preventDefault();
    show(targets[event.key]);
  }
});

updateMode();
show(current, false);
