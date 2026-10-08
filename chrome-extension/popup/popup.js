const BACKEND_URL = "http://127.0.0.1:8000";

document.addEventListener("DOMContentLoaded", async () => {
  // Check backend health
  try {
    const res = await fetch(`${BACKEND_URL}/health`);
    if (res.ok) {
      document.getElementById("backendBadge").innerText = "API Connected";
      document.getElementById("backendBadge").style.color = "#10b981";
    }
  } catch (e) {
    document.getElementById("backendBadge").innerText = "API Offline";
    document.getElementById("backendBadge").style.color = "#ef4444";
  }

  // Query active tab
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const activeTab = tabs[0];
    if (activeTab && activeTab.url) {
      if (activeTab.url.includes("meet.google.com")) {
        document.getElementById("platformName").innerText = "Google Meet";
      } else if (activeTab.url.includes("zoom.us")) {
        document.getElementById("platformName").innerText = "Zoom Meeting";
      } else if (activeTab.url.includes("teams.microsoft.com")) {
        document.getElementById("platformName").innerText = "Microsoft Teams";
      } else {
        document.getElementById("platformName").innerText = "Standalone / Web";
      }

      // Request captured turn count from content script
      chrome.tabs.sendMessage(activeTab.id, { action: "GET_CAPTURED_COUNT" }, (resp) => {
        if (chrome.runtime.lastError || !resp) {
          document.getElementById("turnCounter").innerText = "Ready";
        } else {
          document.getElementById("turnCounter").innerText = resp.count || 0;
        }
      });
    }
  });

  document.getElementById("openStudioBtn").addEventListener("click", () => {
    chrome.tabs.create({ url: "http://127.0.0.1:8000" });
  });

  document.getElementById("summarizeBtn").addEventListener("click", () => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { action: "TRIGGER_SUMMARIZE" });
    });
  });
});
