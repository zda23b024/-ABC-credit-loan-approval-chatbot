// Small DOM harness for exercising the real browser script without opening a browser.
// The API response is stubbed here; the companion Python API tests exercise inference.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.dataset = {};
    this.style = {};
    this.value = '';
    this.disabled = false;
    this.checked = false;
    this.scrollHeight = 0;
    this._text = '';
    this._className = '';
    this.parentNode = null;
    this.classList = {
      add: name => this.setClass(name, true),
      remove: name => this.setClass(name, false),
      contains: name => this._className.split(/\s+/).includes(name),
      toggle: (name, force) => {
        const active = force === undefined ? !this.classList.contains(name) : force;
        this.setClass(name, active); return active;
      },
    };
  }
  set className(value) { this._className = value; }
  get className() { return this._className; }
  setClass(name, active) {
    const names = new Set(this._className.split(/\s+/).filter(Boolean));
    if (active) names.add(name); else names.delete(name);
    this._className = [...names].join(' ');
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(child => child.textContent).join(''); }
  appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
  append(...items) { for (const item of items) this.appendChild(typeof item === 'string' ? new TextNode(item) : item); }
  replaceChildren(...items) { this.children = []; this._text = ''; this.append(...items); }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  dispatch(name, event = {}) {
    event.preventDefault ||= () => {};
    for (const callback of this.listeners[name] || []) callback(event);
  }
  click() { this.dispatch('click'); }
  focus() {}
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(item => item !== this); }
  querySelectorAll(selector) {
    const matches = node => selector.startsWith('.')
      ? node.className.split(/\s+/).includes(selector.slice(1))
      : node.tagName === selector.toLowerCase();
    const found = [];
    const visit = node => { for (const child of node.children) { if (matches(child)) found.push(child); visit(child); } };
    visit(this); return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}
class TextNode extends Element { constructor(text) { super('#text'); this._text = text; } }

const elements = Object.fromEntries(['chat','composer','step-label','step-count','progress-fill','restart'].map(id => [id,new Element('div')]));
const document = {
  getElementById: id => elements[id],
  createElement: tag => new Element(tag),
  createTextNode: text => new TextNode(text),
};
const catalog = {
  employment_types:[{value:'SAL',label:'Salaried'},{value:'SEP',label:'Self-employed'}],
  vehicle_types:[{value:'MC',label:'Motorcycle'},{value:'SC',label:'Scooter'}],
};
let submitted = null;
let decisionRequests = 0;
async function fetchStub(url, options = {}) {
  if (url === '/api/catalog') return {ok:true,json:async()=>catalog};
  assert.equal(url,'/api/decision');
  decisionRequests += 1;
  submitted = JSON.parse(options.body);
  return {ok:true,json:async()=>({
    decision:'APPROVED',decline_probability:.1,approval_probability:.9,
    threshold_declined:.13995426939656921,session_id:'test-session-id',latency_ms:4.2,warnings:[],
  })};
}

const source = fs.readFileSync('chatbot/static/app.js','utf8');
vm.runInNewContext(source,{document,fetch:fetchStub,Intl,Number,Promise,console});
const tick = () => new Promise(resolve => setImmediate(resolve));
const form = () => elements.composer.children.find(node => node.tagName === 'form');
const choose = (value) => {
  const button = form().querySelectorAll('.choice-button').find(node => node.dataset.value === String(value));
  assert.ok(button,`choice ${value} is available`); button.click();
};
const submit = () => form().dispatch('submit');
const editButtons = () => elements.chat.querySelector('.review-message').querySelectorAll('.edit-button');
const currentQuestionForm = () => {assert.ok(form(),'question form is shown');return form();};

(async () => {
  await tick();
  const consent = form();
  consent.querySelector('input').checked = true;
  submit();

  // Answer age, go back from question 2, and return without losing the answer.
  form().querySelector('.input-control').value='30';
  submit();
  currentQuestionForm().querySelector('.secondary-button').click();
  assert.match(elements.chat.textContent,/Edit answer: What is your age/);
  assert.equal(form().querySelector('.input-control').value,'30 years');
  submit();
  choose('SAL'); submit();
  form().querySelector('.input-control').value='42000';
  submit();
  choose('MC'); submit();
  form().querySelector('.input-control').value='100000';
  submit();
  form().querySelector('.input-control').value='125000';
  submit();
  choose('no'); submit();

  // The applicant sees all seven answers before the model is called.
  let review = elements.chat.querySelector('.review-message');
  assert.ok(review);
  assert.equal(review.querySelectorAll('.review-item').length,7);
  assert.equal(decisionRequests,0);

  // Edit age from the review and return to the review screen.
  editButtons()[0].click();
  form().querySelector('.input-control').value='36';
  submit();
  review = elements.chat.querySelector('.review-message');
  assert.match(review.textContent,/36 years/);
  assert.match(elements.chat.textContent,/Updated answer: 36 years/);

  // Exact loan and vehicle prices recalculate LTV on review.
  editButtons()[4].click();
  form().querySelector('.input-control').value='180000';
  submit();
  review = elements.chat.querySelector('.review-message');
  assert.match(review.querySelector('.review-calculation').textContent,/144\.00%/);
  assert.equal(review.querySelectorAll('.review-item').length,7);
  assert.equal(decisionRequests,0);

  const confirm = elements.composer.querySelectorAll('.send-button').find(button=>button.textContent==='Confirm and get assessment');
  assert.ok(confirm);confirm.click();await tick();await tick();
  assert.equal(decisionRequests,1);
  assert.equal(submitted.age,36);
  assert.equal(submitted.employment_type,'SAL');
  assert.equal(submitted.monthly_net_income,42000);
  assert.equal(submitted.vehicle_type,'MC');
  assert.equal(submitted.loan_amount,180000);
  assert.equal(submitted.vehicle_price,125000);
  assert.equal(submitted.has_active_loan,false);
  assert.equal(submitted.consent_to_log,true);
  assert.deepEqual(Object.keys(submitted).sort(),[
    'age','consent_to_log','employment_type','has_active_loan',
    'loan_amount','monthly_net_income','vehicle_price','vehicle_type',
  ]);
  assert.match(elements.chat.textContent,/APPROVED/);
  console.log('PASS: 7 exact answers, Back, review/edit, LTV recalculation, confirmed submit.');
})().catch(error=>{console.error(error);process.exitCode=1;});
