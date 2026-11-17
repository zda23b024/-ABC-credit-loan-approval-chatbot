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
  qualifications:[{value:'GRAD',label:'Graduate'},{value:'SSC',label:'Secondary'}],
  employment_types:[{value:'SAL',label:'Salaried'}],
  residences:[{value:'O',label:'Owned'}],
  area_tiers:[{value:'01 Megapolis (A)',label:'Megapolis A'}],
  vehicle_types:[{value:'MC',label:'Motorcycle'},{value:'SC',label:'Scooter'}],
  vehicles:[
    {description:'Roadster MC',product_code:'MC',make_code:'Maker M'},
    {description:'Commuter MC',product_code:'MC',make_code:'Maker M'},
    {description:'Scoot SC',product_code:'SC',make_code:'Maker S'},
  ],
  representative_values:{
    age_band:{'18-25':22,'26-35':30,'36-45':40,'46-60':53},
    income_band:{'0':0,under_15000:10000},
    loan_band:{under_50000:40000,'200000_plus':225000},
    vehicle_price_band:{under_75000:65000,'75000_94999':85000,'155000_174999':165000,'175000_plus':200000},
  },
};
let submitted = null;
let decisionRequests = 0;
async function fetchStub(url, options = {}) {
  if (url === '/api/catalog') return {ok:true,json:async()=>catalog};
  if (url === '/api/health') return {ok:true,json:async()=>({deployment_mode:'local',persistent_audit_logging:true})};
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

  // Answer question 1, use Back from question 2, and return without losing progress.
  choose('18-25'); submit();
  currentQuestionForm().querySelector('.secondary-button').click();
  assert.match(elements.chat.textContent,/Edit answer: What is the applicant/);
  assert.equal(form().querySelector('.input-control').value,'18–25 years');
  submit();
  choose('GRAD'); submit();
  choose('SAL'); submit();
  choose('0'); submit();
  choose('O'); submit();
  choose('01 Megapolis (A)'); submit();
  choose('MC'); submit();
  form().querySelector('select').value='Roadster MC';
  form().querySelector('select').dispatch('change'); submit();
  choose('under_50000'); submit();
  choose('under_75000'); submit();
  choose('no'); submit();

  // The applicant sees all 11 answers before the model is called.
  let review = elements.chat.querySelector('.review-message');
  assert.ok(review);
  assert.equal(review.querySelectorAll('.review-item').length,11);
  assert.equal(decisionRequests,0);

  // Edit age from the review and return to the review screen.
  editButtons()[0].click(); choose('36-45'); submit();
  review = elements.chat.querySelector('.review-message');
  assert.match(review.textContent,/36–45 years/);
  assert.match(elements.chat.textContent,/Updated answer: 36–45 years/);

  // Changing vehicle type invalidates the old model and asks for a matching one.
  editButtons()[6].click(); choose('SC'); submit();
  assert.match(elements.chat.textContent,/Choose the vehicle model/);
  assert.deepEqual(form().querySelector('select').children.map(option=>option.value),['','Scoot SC']);
  form().querySelector('select').value='Scoot SC';form().querySelector('select').dispatch('change');submit();
  review = elements.chat.querySelector('.review-message');
  assert.match(review.textContent,/Scoot SC/);

  // Changing loan amount invalidates vehicle price; the new LTV is recalculated.
  editButtons()[8].click(); choose('200000_plus'); submit();
  assert.match(elements.chat.textContent,/Approximate total vehicle price/);
  assert.deepEqual(form().querySelectorAll('.choice-button').map(button=>button.dataset.value),['155000_174999','175000_plus']);
  choose('175000_plus');submit();
  review = elements.chat.querySelector('.review-message');
  assert.match(review.querySelector('.review-calculation').textContent,/112\.50%/);
  assert.equal(review.querySelectorAll('.review-item').length,11);
  assert.equal(decisionRequests,0);

  const confirm = elements.composer.querySelectorAll('.send-button').find(button=>button.textContent==='Confirm and get assessment');
  assert.ok(confirm);confirm.click();await tick();await tick();
  assert.equal(decisionRequests,1);
  assert.equal(submitted.age_band,'36-45');
  assert.equal(submitted.qualification,'GRAD');
  assert.equal(submitted.vehicle_type,'SC');
  assert.equal(submitted.model_description,'Scoot SC');
  assert.equal(submitted.loan_band,'200000_plus');
  assert.equal(submitted.vehicle_price_band,'175000_plus');
  assert.equal(submitted.has_active_loan,false);
  assert.equal(submitted.consent_to_log,true);
  assert.match(elements.chat.textContent,/APPROVED/);
  console.log('PASS: 11 answers, Back, review/edit, dependent vehicle and price updates, LTV recalculation, confirmed submit.');
})().catch(error=>{console.error(error);process.exitCode=1;});
