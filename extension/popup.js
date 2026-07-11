// Load saved API URL
document.addEventListener('DOMContentLoaded', () => {
  chrome.storage.sync.get(['apiUrl'], (result) => {
    if (result.apiUrl) {
      document.getElementById('apiUrl').value = result.apiUrl;
    }
  });
});

// Save API URL
document.getElementById('saveBtn').addEventListener('click', () => {
  const apiUrl = document.getElementById('apiUrl').value.trim();
  if (apiUrl) {
    chrome.storage.sync.set({ apiUrl }, () => {
      const btn = document.getElementById('saveBtn');
      btn.textContent = 'Saved! ✅';
      btn.style.background = '#10b981';
      setTimeout(() => {
        btn.textContent = 'Save API URL';
        btn.style.background = '#3b82f6';
      }, 2000);
    });
  }
});