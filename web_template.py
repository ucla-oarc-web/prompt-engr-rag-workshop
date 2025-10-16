import streamlit as st

css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    //display: flex
}
.chat-message.user {
    background-color: white;
    position: relative;
    left: 75%;
    width: 25%;
    border-bottom-right-radius: unset !important;
    border-radius: 20px;
}
.chat-message.bot {
    background-color: white;
    position: relative;
    left: 0;
    width: 70%;
    border-radius: 20px;
}

.chat-message.bot button {
    background-color: white;
    border-radius: 0.8rem;
}

.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  //padding: 0 1.5rem;
  color: black;
}

.chat-message .message ol, p {
  color: black;  
}

.page-title-wrapper {
    text-align:center;
}
.page-title-wrapper h1{
    font-family: cursive;
}
.page-title-wrapper h4{
    color: black;
}
.hr-sidebar {
  display: block;
  height: 1px;
  border: 0;
  border-top: 2px solid white;
  margin: 1em 0;
  padding: 0;
}
.notice-text {
    color: black;
    text-align: center;
    font-size: 0.9rem;
    padding-bottom: 1rem;
    position: fixed;
    bottom: 0px;
    z-index: 1000;
    margin: 0 auto;
    width: 70%;
}
#sidebar-about,#sidebar-howto {
    color: white;
}

.wrap-collabsible {
  margin-bottom: 0.4rem;
}

input[type='checkbox'] {
  display: none;
}

.lbl-toggle, .lbl-toggle2 {
  display: block;

  font-weight: 600;
  font-size: 1rem;
  text-align: center;

  padding: 0.3rem;

  color: black;
  background: white;

  cursor: pointer;

  border-radius: 7px;
  transition: all 0.25s ease-out;
}

.lbl-toggle:hover, .lbl-toggle2:hover {
  color: #003B5C;
}

.lbl-toggle::before, .lbl-toggle2::before {
  content: ' ';
  display: inline-block;

  border-top: 5px solid transparent;
  border-bottom: 5px solid transparent;
  border-left: 5px solid currentColor;
  vertical-align: middle;
  margin-right: .7rem;
  transform: translateY(-2px);

  transition: transform .2s ease-out;
}

.toggle:checked + .lbl-toggle::before {
  transform: rotate(90deg) translateX(-3px);
}

.toggle2:checked + .lbl-toggle2::before {
  transform: rotate(90deg) translateX(-3px);
}

.collapsible-content, .collapsible-content2 {
  max-height: 0px;
  overflow: hidden;
  transition: max-height .25s ease-in-out;
}

.toggle:checked + .lbl-toggle + .collapsible-content {
  max-height: 100vh;
}

.toggle2:checked + .lbl-toggle2 + .collapsible-content2 {
  max-height: 100vh;
}

.toggle:checked + .lbl-toggle {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}

.toggle2:checked + .lbl-toggle2 {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}

.collapsible-content .content-inner, .collapsible-content2 .content-inner {
  background: #003B5C;
  border-bottom: 1px solid rgba(250, 224, 66, .45);
  border-bottom-left-radius: 7px;
  border-bottom-right-radius: 7px;
  padding: .5rem 1rem;
}

.collapsible-content .content-inner p, .collapsible-content2 .content-inner p  {
  color: white;  
}

.rag-icons img {
  background-color: white;
  width: 40px;
  height: 40px;
  margin-right: 0.5rem;
}

.rag-icons button img {
  width: 30px;
  height: 30px;
  margin-right: 0;
}

.rag-icons button{
  position: relative;
  left: 67%;
}
'''

sidebar_about = '''
<div class="wrap-collabsible">
  <input id="collapsible" class="toggle" type="checkbox">
  <label for="collapsible" class="lbl-toggle">About</label>
  <div class="collapsible-content">
    <div class="content-inner">
      <p>This interface allows you to ask questions about your documents and get accurate answers.<br/><br/>
        This site is a work in progress. You can contribute with your feedback and suggestions</p>
    </div>
  </div>
</div>
'''

sidebar_howto = '''
<div class="wrap-collabsible">
  <input id="collapsible2" class="toggle2" type="checkbox">
  <label for="collapsible2" class="lbl-toggle2">How to use</label>
  <div class="collapsible-content2">
    <div class="content-inner">
      <p>
        1. Select a llm model and an embedding model<br/>
        2. Upload a pdf, url, or (txt file, csv, video etc.)<br/>
        3. Ask a question about the document<br/>        
      </p>
    </div>
  </div>
</div>
'''


page_title = '''
<div class="page-title-wrapper">
    <h1>BruinRag</h1>
</div>
'''

script_save_to_note = '''
<script>
function copyParentText(button) {
  // Get the parent element of the button
  const parentElement = document.getElementsByClassName("example");

  // Get the text content of the parent element
  const parentText = parentElement.textContent;

  // Get the target element where you want to add the text
  const targetElement = document.getElementById('abcd);

  // Append the parent text to the target element
  targetElement.textContent += parentText;
}
</script>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
    <div class="rag-icons" id="abcd">
        <img src="data:image/png;base64,{{IMG1}}"> 
        <img src="data:image/png;base64,{{IMG2}}"> 
        <img src="data:image/png;base64,{{IMG3}}"> 
    </div>
</div>
'''

user_template = '''
<div class="chat-message user">   
    <div class="message">{{MSG}}</div>
</div>
'''

hr_sidebar = '''
<hr class='hr-sidebar'/>
'''

notice_text = '''
<p class="notice-text">
Responses are generated by an LLM and may contain errors.
</p>
'''