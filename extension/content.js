// API URL — update this when you deploy to Render
const API_URL = "https://unloving-relation-thumping.ngrok-free.dev";

// Track processed messages to avoid duplicates
const processedMessages = new Set();

// Main function to check a response
async function checkHallucination(question, aiAnswer, badgeContainer) {
  try {
    badgeContainer.innerHTML = `
      <div style="
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 12px; border-radius: 20px;
        background: #f3f4f6; border: 1px solid #d1d5db;
        font-size: 12px; color: #6b7280; margin-top: 8px;
        font-family: sans-serif;
      ">
        ⏳ Checking for hallucinations...
      </div>
    `;

    const response = await fetch(`${API_URL}/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, ai_answer: aiAnswer })
    });

    if (!response.ok) throw new Error("API error");

    const data = await response.json();
    showBadge(badgeContainer, data);

  } catch (error) {
    badgeContainer.innerHTML = `
      <div style="
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 12px; border-radius: 20px;
        background: #f3f4f6; border: 1px solid #d1d5db;
        font-size: 12px; color: #6b7280; margin-top: 8px;
        font-family: sans-serif;
      ">
        ⚠️ Could not verify (API unavailable)
      </div>
    `;
  }
}

// Show result badge
function showBadge(container, data) {
  const colors = {
    FACTUAL: { bg: '#d1fae5', border: '#6ee7b7', text: '#065f46', emoji: '✅' },
    HALLUCINATION: { bg: '#fee2e2', border: '#fca5a5', text: '#991b1b', emoji: '⚠️' },
    UNCERTAIN: { bg: '#fef3c7', border: '#fcd34d', text: '#92400e', emoji: '❓' }
  };

  const c = colors[data.label] || colors.UNCERTAIN;
  const confidence = Math.round(data.confidence * 100);
  const evidence = data.evidence?.[0]?.text?.substring(0, 100) || '';

  container.innerHTML = `
    <div style="
      margin-top: 10px; padding: 10px 14px;
      border-radius: 10px; border: 1px solid ${c.border};
      background: ${c.bg}; font-family: sans-serif;
      max-width: 600px;
    ">
      <div style="
        font-size: 13px; font-weight: 600;
        color: ${c.text}; margin-bottom: 4px;
      ">
        ${c.emoji} ${data.label} — ${confidence}% confidence
      </div>
      <div style="font-size: 11px; color: ${c.text}; opacity: 0.8;">
        ${data.summary}
      </div>
      ${evidence ? `
        <div style="
          font-size: 11px; color: ${c.text};
          opacity: 0.7; margin-top: 4px;
          border-top: 1px solid ${c.border}; padding-top: 4px;
        ">
          📄 Evidence: ${evidence}...
        </div>
      ` : ''}
    </div>
  `;
}

// Find ChatGPT responses and inject badges
function processResponses() {
  // ChatGPT response selector
  const responses = document.querySelectorAll('[data-message-author-role="assistant"]');
  const questions = document.querySelectorAll('[data-message-author-role="user"]');

  responses.forEach((response, index) => {
    const responseText = response.innerText?.trim();
    if (!responseText || responseText.length < 20) return;

    // Avoid duplicate processing
    const msgId = responseText.substring(0, 50);
    if (processedMessages.has(msgId)) return;
    processedMessages.add(msgId);

    // Get corresponding question
    const questionText = questions[index]?.innerText?.trim() || "Unknown question";

    // Create badge container
    const badgeContainer = document.createElement('div');
    badgeContainer.className = 'hallucination-badge';
    response.appendChild(badgeContainer);

    // Check hallucination
    checkHallucination(questionText, responseText, badgeContainer);
  });
}

// Watch for new responses using MutationObserver
const observer = new MutationObserver((mutations) => {
  for (const mutation of mutations) {
    if (mutation.addedNodes.length > 0) {
      setTimeout(processResponses, 2000);
    }
  }
});

// Start observing
observer.observe(document.body, {
  childList: true,
  subtree: true
});

// Run on page load
setTimeout(processResponses, 3000);
console.log("🔍 Hallucination Detector active!");