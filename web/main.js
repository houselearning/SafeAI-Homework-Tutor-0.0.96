const btn = document.getElementById("send");
const promptEl = document.getElementById("prompt");
const answerEl = document.getElementById("answer");

// Set this to your deployed backend URL
const API_URL = "https://your-backend.example.com/generate";

btn.onclick = async () => {
  const prompt = promptEl.value.trim();
  if (!prompt) return;
  answerEl.textContent = "Thinking...";
  const res = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, max_new_tokens: 128 }),
  });
  const data = await res.json();
  answerEl.textContent = data.completion;
};
