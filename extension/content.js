// API URL — update this when you deploy to Render. For local testing, change to http://localhost:8000
const API_URL = "https://hallucination-detector-b9jx.onrender.com";

// Track processed messages to avoid duplicates
const processedMessages = new Set();

// Helper to fetch with timeout
async function fetchWithTimeout(url, options = {}, timeout = 120000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal
    });
    clearTimeout(timer);
    return response;
  } catch (error) {
    clearTimeout(timer);
    throw error;
  }
}

// Main function to check a response
async function checkHallucination(question, aiAnswer, badgeContainer) {
  // Set initial status
  badgeContainer.innerHTML = `
    <div id="badge-status-container" style="
      display: inline-flex; align-items: center; gap: 6px;
      padding: 6px 12px; border-radius: 20px;
      background: #f3f4f6; border: 1px solid #d1d5db;
      font-size: 12px; color: #4b5563; margin-top: 8px;
      font-family: sans-serif;
    ">
      <span>⏳ Checking for hallucinations...</span>
    </div>
  `;

  // Start a timer to show "warming up" if server takes more than 4 seconds to respond (Render spin-down check)
  const warmupTimer = setTimeout(() => {
    const statusDiv = badgeContainer.querySelector('#badge-status-container');
    if (statusDiv) {
      statusDiv.innerHTML = `<span>⏳ Warming up backend server... (Render free tier takes ~60s to wake up)</span>`;
      statusDiv.style.background = "#fffbeb";
      statusDiv.style.border = "1px solid #fde68a";
      statusDiv.style.color = "#b45309";
    }
  }, 4000);

  try {
    const response = await fetchWithTimeout(`${API_URL}/check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, ai_answer: aiAnswer })
    }, 120000); // 120 seconds timeout

    clearTimeout(warmupTimer);

    if (!response.ok) {
      throw new Error(`API returned HTTP ${response.status}`);
    }

    const data = await response.json();
    showBadge(badgeContainer, data);

  } catch (error) {
    clearTimeout(warmupTimer);
    console.error("Hallucination Detector Error:", error);
    
    let errorMsg = "Could not verify (API unavailable)";
    if (error.name === 'AbortError') {
      errorMsg = "Verification timed out (Server failed to respond in 120s)";
    }

    badgeContainer.innerHTML = `
      <div style="
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 12px; border-radius: 20px;
        background: #fef2f2; border: 1px solid #fca5a5;
        font-size: 12px; color: #b91c1c; margin-top: 8px;
        font-family: sans-serif;
      ">
        ⚠️ ${errorMsg}
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
  const evidence = data.evidence?.[0]?.text?.substring(0, 150) || '';
  const evidenceUrl = data.evidence?.[0]?.url || '';

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
          opacity: 0.7; margin-top: 6px;
          border-top: 1px solid ${c.border}; padding-top: 6px;
        ">
          📄 <strong>Evidence chunk:</strong> "${evidence}..."
          ${evidenceUrl ? `<br/><a href="${evidenceUrl}" target="_blank" style="color: ${c.text}; text-decoration: underline; font-weight: 500; font-size: 10px; display: inline-block; margin-top: 2px;">Read full source page</a>` : ''}
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
    // 1. Avoid duplicates: If this response block already has a badge injected, skip it!
    if (response.querySelector('.hallucination-badge')) return;

    // 2. Ignore messages that are still actively streaming/typing
    if (response.classList.contains('result-streaming')) return;

    const responseText = response.innerText?.trim();
    if (!responseText || responseText.length < 20) return;

    // 3. Prevent duplicate calls using the message ID set
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