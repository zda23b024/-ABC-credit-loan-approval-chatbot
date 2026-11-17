(() => {
  'use strict';

  const chat = document.getElementById('chat');
  const composer = document.getElementById('composer');
  const stepLabel = document.getElementById('step-label');
  const stepCount = document.getElementById('step-count');
  const progressFill = document.getElementById('progress-fill');
  const restartButton = document.getElementById('restart');
  const fields = [
    {key:'age_band', question:'What is the applicant’s age?', type:'choice', helper:'Choose an age range or type the exact age. A range is converted to a representative age for this estimate.', choices:[['18-25','18–25 years'],['26-35','26–35 years'],['36-45','36–45 years'],['46-60','46–60 years']]},
    {key:'qualification', question:'Highest education completed?', type:'choices', source:'qualifications'},
    {key:'employment_type', question:'Main work or income type?', type:'choices', source:'employment_types'},
    {key:'income_band', question:'Approximate monthly take-home income?', type:'choice', helper:'Choose a range or type the exact monthly amount in rupees. A range uses a representative amount for this estimate.', choices:[['0','₹0'],['under_15000','Under ₹15,000'],['15000_24999','₹15,000–₹24,999'],['25000_29999','₹25,000–₹29,999'],['30000_36999','₹30,000–₹36,999'],['37000_49999','₹37,000–₹49,999'],['50000_84999','₹50,000–₹84,999'],['85000_plus','₹85,000 or more']]},
    {key:'residence', question:'Where does the applicant live?', type:'choices', source:'residences'},
    {key:'area_tier', question:'Which broad area type is closest?', type:'choices', source:'area_tiers', helper:'Choose a broad category only. No postal code or address.'},
    {key:'vehicle_type', question:'What type of vehicle is being financed?', type:'choices', source:'vehicle_types'},
    {key:'model_description', question:'Choose the vehicle model.', type:'vehicle', helper:'The list is filtered by vehicle type. Select a model, then press Send.'},
    {key:'loan_band', question:'Approximate loan amount requested?', type:'choice', helper:'Choose a range or type the exact requested amount in rupees. A range uses a representative amount for this estimate.', choices:[['under_50000','Under ₹50,000'],['50000_74999','₹50,000–₹74,999'],['75000_99999','₹75,000–₹99,999'],['100000_124999','₹1,00,000–₹1,24,999'],['125000_149999','₹1,25,000–₹1,49,999'],['150000_199999','₹1,50,000–₹1,99,999'],['200000_plus','₹2,00,000 or more']]},
    {key:'vehicle_price_band', question:'Approximate total vehicle price?', type:'choice', helper:'Choose a range or type the exact price. A range uses a representative price to calculate LTV.', choices:[['under_75000','Under ₹75,000'],['75000_94999','₹75,000–₹94,999'],['95000_114999','₹95,000–₹1,14,999'],['115000_129999','₹1,15,000–₹1,29,999'],['130000_154999','₹1,30,000–₹1,54,999'],['155000_174999','₹1,55,000–₹1,74,999'],['175000_plus','₹1,75,000 or more']]},
    {key:'has_active_loan', question:'Does the applicant currently have an active prior loan?', type:'choice', choices:[['yes','Yes, an active prior loan'],['no','No active prior loan']]},
  ];
  const TOTAL = fields.length;
  let catalog = null;
  let current = 0;
  let answers = {};
  let savedResponses = {};
  let busy = false;
  let editResumeIndex = null;
  let syntheticDemo = false;

  function hasOwn(object, key) { return Object.prototype.hasOwnProperty.call(object, key); }
  function fieldAnswered(field) {
    if (field.key === 'age_band') return hasOwn(answers, 'age_band') || hasOwn(answers, 'age_value');
    if (field.key === 'income_band') return hasOwn(answers, 'income_band') || hasOwn(answers, 'monthly_net_income');
    if (field.key === 'loan_band') return hasOwn(answers, 'loan_band') || hasOwn(answers, 'loan_amount');
    if (field.key === 'vehicle_price_band') return hasOwn(answers, 'vehicle_price_band') || hasOwn(answers, 'vehicle_price');
    return hasOwn(answers, field.key);
  }
  function firstUnansweredIndex() { return fields.findIndex(field => !fieldAnswered(field)); }
  function updateProgress() {
    const count = fields.filter(fieldAnswered).length;
    stepCount.textContent = `${count} / ${TOTAL}`;
    progressFill.style.width = `${count / TOTAL * 100}%`;
    stepLabel.textContent = current >= TOTAL ? 'Review answers' : count === 0 ? 'Ready when you are' : 'Application details';
  }
  function addMessage(role, text) {
    const row = document.createElement('div'); row.className = `message ${role}`;
    const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = text;
    row.appendChild(bubble); chat.appendChild(row); chat.scrollTop = chat.scrollHeight; return row;
  }
  function clearComposer() { composer.replaceChildren(); }
  function setError(form, text) {
    let error = form.querySelector('.error');
    if (!error) { error = document.createElement('p'); error.className = 'error'; error.setAttribute('role', 'alert'); form.appendChild(error); }
    error.textContent = text;
  }
  function choicesFor(field) {
    if (field.type === 'choices') return (catalog[field.source] || []).map(item => [item.value, item.label]);
    if (field.type === 'vehicle') return catalog.vehicles.filter(v => v.product_code === answers.vehicle_type).map(v => [v.description, `${v.description} · ${v.make_code}`]);
    if (field.key === 'vehicle_price_band') {
      const loan = answers.loan_amount ?? catalog.representative_values.loan_band[answers.loan_band];
      if (!loan) return field.choices;
      return field.choices.filter(([key]) => 100 * loan / catalog.representative_values.vehicle_price_band[key] <= 150);
    }
    return field.choices || [];
  }
  function answerForTyped(field, text) {
    const clean = text.trim();
    if (!clean) return {error:'Choose an option above or type an answer.'};
    if (field.source) {
      const match = (catalog[field.source] || []).find(x => x.value.toLowerCase() === clean.toLowerCase() || x.label.toLowerCase() === clean.toLowerCase());
      return match ? {payload:{[field.key]:match.value}, display:match.label} : {error:'Please choose or type one of the listed options.'};
    }
    if (field.type === 'vehicle') {
      const match = catalog.vehicles.find(v => v.product_code === answers.vehicle_type && (v.description.toLowerCase() === clean.toLowerCase() || `${v.description} · ${v.make_code}`.toLowerCase() === clean.toLowerCase()));
      return match ? {payload:{model_description:match.description}, display:`${match.description} · ${match.make_code}`} : {error:'Please choose a model from the list for the selected vehicle type.'};
    }
    const choice = (field.choices || []).find(([value,label]) => value.toLowerCase() === clean.toLowerCase() || label.toLowerCase() === clean.toLowerCase());
    if (choice) {
      if (field.key === 'has_active_loan') return {payload:{has_active_loan:choice[0] === 'yes'}, display:choice[1]};
      return {payload:{[field.key]:choice[0]}, display:choice[1]};
    }
    if (['age_band','income_band','loan_band','vehicle_price_band'].includes(field.key)) {
      const numeric = clean.replace(/[₹,\s]/g, '').replace(/INR/gi, '');
      if (!/^\d+(\.\d+)?$/.test(numeric)) return {error:'Type a number or choose one of the suggested ranges.'};
      const n = Number(numeric);
      if (!Number.isFinite(n)) return {error:'Please enter a valid number.'};
      if (field.key === 'age_band') {
        if (!Number.isInteger(n) || n < 18 || n > 60) return {error:'Age must be a whole number from 18 to 60.'};
        return {payload:{age_value:n}, display:`${n} years`};
      }
      if (field.key === 'income_band') {
        if (n < 0 || n > 10000000) return {error:'Enter monthly income from ₹0 to ₹10,000,000.'};
        return {payload:{monthly_net_income:n}, display:`₹${new Intl.NumberFormat('en-IN').format(n)} exact`};
      }
      if (field.key === 'loan_band') {
        if (n <= 0 || n > 50000000) return {error:'Enter a loan amount above ₹0 and no more than ₹50,000,000.'};
        return {payload:{loan_amount:n}, display:`₹${new Intl.NumberFormat('en-IN').format(n)} exact`};
      }
      if (n <= 0 || n > 100000000) return {error:'Enter a vehicle price above ₹0 and no more than ₹100,000,000.'};
      return {payload:{vehicle_price:n}, display:`₹${new Intl.NumberFormat('en-IN').format(n)} exact`};
    }
    return {error:'Please choose one of the listed answers.'};
  }
  function removeAnswer(fieldKey) {
    const alternatives = {
      age_band:['age_value'], income_band:['monthly_net_income'],
      loan_band:['loan_amount'], vehicle_price_band:['vehicle_price'],
    };
    delete answers[fieldKey];
    for (const key of alternatives[fieldKey] || []) delete answers[key];
    delete savedResponses[fieldKey];
  }
  function numericLoan(source = answers) {
    return source.loan_amount ?? catalog.representative_values.loan_band[source.loan_band] ?? null;
  }
  function numericVehiclePrice(source = answers) {
    return source.vehicle_price ?? catalog.representative_values.vehicle_price_band[source.vehicle_price_band] ?? null;
  }
  function storeAnswer(field, result) {
    const oldVehicleType = answers.vehicle_type;
    const oldLoan = numericLoan();
    removeAnswer(field.key);
    Object.assign(answers, result.payload);
    savedResponses[field.key] = {payload:{...result.payload}, display:result.display || result.label};
    if (field.key === 'vehicle_type' && oldVehicleType !== answers.vehicle_type) removeAnswer('model_description');
    if (field.key === 'loan_band' && oldLoan !== numericLoan()) removeAnswer('vehicle_price_band');
  }
  function begin() {
    clearComposer();
    addMessage('assistant','Hi! I’ll ask 11 quick questions. Tap a suggestion above or type your answer below, then press Send.');
    addMessage('assistant',syntheticDemo?'This hosted assessment is a synthetic-data demonstration only. Use fictional answers; never enter real applicant information. It reflects past APPROVED and DECLINED labels, not repayment ability, and results are not stored in persistent audit storage.':'Your assessment is based on patterns in earlier APPROVED and DECLINED applications. It does not predict repayment or guarantee approval; ABC Credit confirms applications through its official review. Please don’t enter names, phone numbers, or exact addresses.');
    const form = document.createElement('form'); const line = document.createElement('label'); line.className = 'checkline';
    const check = document.createElement('input'); check.type = 'checkbox'; check.required = true;
    const note = document.createElement('span'); note.textContent = syntheticDemo?'I will use fictional answers only and understand this hosted demonstration does not persist submissions.':'I understand my answers and assessment result will be recorded without names or contact details.'; line.append(check,note);
    const send = document.createElement('button'); send.className = 'send-button'; send.type = 'submit'; send.textContent = 'Start';
    const row = document.createElement('div'); row.className = 'control-row'; row.appendChild(send); form.append(line,row);
    form.addEventListener('submit',e => {e.preventDefault(); if (!check.checked) return setError(form,'Please check the notice to continue.'); showQuestion();});
    composer.appendChild(form); updateProgress();
  }
  function navigateBack() {
    if (editResumeIndex === null) editResumeIndex = current;
    if (current > 0) { current -= 1; showQuestion(); }
  }
  function editQuestion(index) {
    chat.querySelectorAll('.review-message').forEach(message => message.remove());
    editResumeIndex = TOTAL; current = index; showQuestion();
  }
  function showQuestion() {
    if (current >= TOTAL) return showReview();
    updateProgress(); const field = fields[current];
    addMessage('assistant', editResumeIndex === null ? field.question : `Edit answer: ${field.question}`);
    clearComposer();
    const form = document.createElement('form'); const options = document.createElement('div'); options.className = 'choice-chips';
    const input = document.createElement('input'); input.className = 'input-control'; input.type = 'text'; input.autocomplete = 'off'; input.placeholder = 'Type answer or choose above'; input.setAttribute('aria-label','Type an answer');
    let selected = null;
    const send = document.createElement('button'); send.className = 'send-button'; send.type = 'submit'; send.textContent = 'Save answer';
    send.disabled = true;
    function choose(value,label,payload) {
      selected = {value,label,payload}; input.value = label;
      options.querySelectorAll('.choice-button').forEach(button => {
        const active = button.dataset.value === String(value); button.classList.toggle('selected',active); button.setAttribute('aria-pressed',active ? 'true' : 'false');
      });
      send.disabled = false;
    }
    const list = choicesFor(field); const saved = savedResponses[field.key];
    if (field.type === 'vehicle') {
      const select = document.createElement('select'); select.className = 'select-control';
      const placeholder = document.createElement('option'); placeholder.value = ''; placeholder.textContent = 'Choose a model'; placeholder.disabled = true; select.appendChild(placeholder);
      for (const [value,label] of list) {const option=document.createElement('option');option.value=value;option.textContent=label;select.appendChild(option);}
      select.addEventListener('change',() => {const item=list.find(x=>x[0]===select.value);if(item)choose(item[0],item[1],{model_description:item[0]});});
      options.appendChild(select);
      if (saved && list.some(x => x[0] === saved.payload.model_description)) {
        select.value = saved.payload.model_description;
        const item = list.find(x => x[0] === select.value); choose(item[0],item[1],saved.payload);
      }
    } else {
      for (const [value,label] of list) {
        const option=document.createElement('button');option.type='button';option.className='choice-button';option.textContent=label;option.dataset.value=value;option.setAttribute('aria-pressed','false');
        option.addEventListener('click',() => {
          let payload;
          if (field.type === 'choices') payload = {[field.key]:value};
          else if (field.key === 'has_active_loan') payload = {has_active_loan:value === 'yes'};
          else payload = {[field.key]:value};
          choose(value,label,payload);
        }); options.appendChild(option);
      }
      if (saved) choose(saved.payload[field.key] ?? saved.payload.age_value ?? saved.payload.monthly_net_income ?? saved.payload.loan_amount ?? saved.payload.vehicle_price ?? String(saved.payload.has_active_loan),saved.display,saved.payload);
    }
    if (field.helper) {const helper=document.createElement('p');helper.className='helper';helper.textContent=field.helper;form.append(options,helper);} else form.appendChild(options);
    input.addEventListener('input',() => {selected=null;options.querySelectorAll('.choice-button').forEach(button=>{button.classList.remove('selected');button.setAttribute('aria-pressed','false');});send.disabled=!input.value.trim();});
    const row=document.createElement('div');row.className='control-row composer-entry';
    if (current > 0) {const back=document.createElement('button');back.className='secondary-button';back.type='button';back.textContent='Back';back.setAttribute('aria-label','Go to previous question');back.addEventListener('click',navigateBack);row.appendChild(back);}
    row.append(input,send);form.appendChild(row);
    form.addEventListener('submit',e=>{
      e.preventDefault();let result=selected;
      if (!result || input.value !== result.label) result=answerForTyped(field,input.value);
      if (result.error) return setError(form,result.error);
      const changingPriorAnswer = editResumeIndex !== null;
      storeAnswer(field,result);addMessage('user',changingPriorAnswer?`Updated answer: ${result.display||result.label}`:(result.display||result.label));
      if (editResumeIndex !== null) {
        const resumeAt=editResumeIndex; editResumeIndex=null;
        const missing=firstUnansweredIndex();current=missing>=0?missing:resumeAt;
        if (current>=TOTAL) showReview(); else showQuestion();
      } else {
        current += 1;
        while (current<TOTAL && fieldAnswered(fields[current])) current += 1;
        if (current>=TOTAL) showReview(); else showQuestion();
      }
    });
    composer.appendChild(form);input.focus({preventScroll:true});chat.scrollTop=chat.scrollHeight;
  }
  function displayFor(field) {return savedResponses[field.key]?.display || 'Not answered';}
  function formatCurrency(value) {return `₹${new Intl.NumberFormat('en-IN',{maximumFractionDigits:2}).format(value)}`;}
  function showReview() {
    current=TOTAL;updateProgress();clearComposer();
    const previous=chat.querySelector('.review-message');if(previous)previous.remove();
    const message=document.createElement('div');message.className='message assistant review-message';
    const bubble=document.createElement('div');bubble.className='bubble review-bubble';
    const title=document.createElement('strong');title.className='review-title';title.textContent='Review your 11 answers';bubble.appendChild(title);
    const list=document.createElement('div');list.className='review-list';
    fields.forEach((field,index)=>{
      const item=document.createElement('div');item.className='review-item';
      const copy=document.createElement('div');copy.className='review-copy';
      const question=document.createElement('span');question.className='review-question';question.textContent=`${index+1}. ${field.question}`;
      const answer=document.createElement('strong');answer.className='review-answer';answer.textContent=displayFor(field);
      copy.append(question,answer);
      const edit=document.createElement('button');edit.className='edit-button';edit.type='button';edit.textContent='Edit';edit.setAttribute('aria-label',`Edit answer ${index+1}`);edit.addEventListener('click',()=>editQuestion(index));
      item.append(copy,edit);list.appendChild(item);
    });
    bubble.appendChild(list);
    const loan=numericLoan(),price=numericVehiclePrice();
    if (loan !== null && price !== null && price > 0) {
      const ltv=document.createElement('p');ltv.className='review-calculation';ltv.textContent=`Calculated LTV: ${(100*loan/price).toFixed(2)}% (${formatCurrency(loan)} loan ÷ ${formatCurrency(price)} vehicle price)`;bubble.appendChild(ltv);
    }
    message.appendChild(bubble);chat.appendChild(message);chat.scrollTop=chat.scrollHeight;
    const note=document.createElement('p');note.className='helper';note.textContent='Check every answer. Use Edit to change one; dependent answers are refreshed where needed.';
    const row=document.createElement('div');row.className='control-row review-actions';
    const back=document.createElement('button');back.className='secondary-button';back.type='button';back.textContent='Back to last question';back.addEventListener('click',()=>editQuestion(TOTAL-1));
    const confirm=document.createElement('button');confirm.className='send-button';confirm.type='button';confirm.textContent='Confirm and get assessment';confirm.addEventListener('click',sendApplication);
    row.append(back,confirm);composer.append(note,row);
  }
  async function sendApplication() {
    if (busy) return;
    const missing=firstUnansweredIndex();
    if (missing>=0) {current=missing;addMessage('assistant','Please complete every answer before continuing.');return showQuestion();}
    const loan=numericLoan(),price=numericVehiclePrice(),ltv=100*loan/price;
    if (!Number.isFinite(ltv)||ltv<0.01||ltv>150) {
      addMessage('assistant','These amounts produce an LTV outside the allowed range. Please edit the loan amount or vehicle price.');
      return showReview();
    }
    busy=true;updateProgress();clearComposer();addMessage('assistant','All set. I’m checking the confirmed answers with the saved model.');
    const loading=document.createElement('p');loading.className='helper';loading.textContent='Generating the result…';composer.appendChild(loading);
    const payload={...answers,has_active_loan:answers.has_active_loan,consent_to_log:true};
    try {
      const response=await fetch('/api/decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const data=await response.json();if(!response.ok)throw new Error(data.detail||'Could not complete this assessment.');showResult(data);
    } catch(error) {
      addMessage('assistant',`I couldn’t complete the assessment: ${error.message}`);
      const retry=document.createElement('button');retry.className='send-button';retry.type='button';retry.textContent='Try again';retry.addEventListener('click',sendApplication);composer.replaceChildren(retry);
    } finally {busy=false;}
  }
  function showResult(data) {
    addMessage('assistant','Your assessment is complete.');const card=document.createElement('div');card.className=`result-card${data.decision==='DECLINED'?' declined':''}`;
    const top=document.createElement('div');top.className='result-top';const left=document.createElement('div');
    const label=document.createElement('div');label.className='result-label';label.textContent='Application assessment';
    const decision=document.createElement('div');decision.className='result-decision';decision.textContent=data.decision;left.append(label,decision);
    const prob=document.createElement('div');prob.className='probability';const value=document.createElement('strong');value.textContent=`${(data.decline_probability*100).toFixed(1)}%`;prob.append(value,document.createTextNode(' decline probability'));top.append(left,prob);
    const copy=document.createElement('p');copy.className='result-copy';copy.textContent='This assessment reflects patterns in past loan decisions. It is not a loan offer and does not predict repayment. Final eligibility is subject to ABC Credit’s official review.';
    const precise=document.createElement('div');precise.className='result-precise';precise.textContent=`Exact decline probability: ${data.decline_probability.toFixed(8)} · threshold: ${data.threshold_declined.toFixed(8)}`;
    const nextStep=document.createElement('p');nextStep.className='result-next-step';nextStep.textContent=data.decision==='DECLINED'?'Next step: ABC Credit can explain the official review outcome and any available options.':'Next step: ABC Credit will confirm eligibility and any offer through its official review.';
    const meta=document.createElement('div');meta.className='result-meta';meta.textContent=`Reference ${data.session_id.slice(0,8)} · Assessment completed in ${data.latency_ms} ms`;
    card.append(top,copy,nextStep,precise,meta);if(data.persistent_audit_logging===false){const warning=document.createElement('div');warning.className='result-warning';warning.textContent='Synthetic demonstration only. This assessment was not written to persistent audit storage.';card.appendChild(warning);}for(const warningText of data.warnings||[]){const warning=document.createElement('div');warning.className='result-warning';warning.textContent=warningText;card.appendChild(warning);}
    const row=document.createElement('div');row.className='message assistant';row.appendChild(card);chat.appendChild(row);chat.scrollTop=chat.scrollHeight;
    clearComposer();const again=document.createElement('button');again.className='send-button';again.type='button';again.textContent='Start again';again.addEventListener('click',restart);composer.appendChild(again);
  }
  function restart() {current=0;answers={};savedResponses={};busy=false;editResumeIndex=null;chat.replaceChildren();begin();}
  async function boot() {try{const [catalogResponse,healthResponse]=await Promise.all([fetch('/api/catalog'),fetch('/api/health')]);if(!catalogResponse.ok||!healthResponse.ok)throw new Error();catalog=await catalogResponse.json();const health=await healthResponse.json();syntheticDemo=health.deployment_mode==='synthetic_demo';if(syntheticDemo){const notice=document.querySelector('.model-note');if(notice)notice.textContent='Hosted synthetic-data demonstration · do not enter real applicant information';const foot=document.querySelector('.card-foot');if(foot)foot.innerHTML='<span class="foot-lock" aria-hidden="true">ⓘ</span> Synthetic inputs only · submissions are not persistently logged';}begin();}catch{addMessage('assistant','Could not load answer choices. Please reload the page.');}}
  restartButton.addEventListener('click',restart);updateProgress();boot();
})();
