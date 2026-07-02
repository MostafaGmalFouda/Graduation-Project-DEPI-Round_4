console.log("Chatbot Loaded");

// ==========================
// Detect a REAL browser refresh (F5 / reload button / Ctrl+R)
// vs. normal navigation between pages of the app.
// Only a genuine reload should wipe the chat / server-side session.
// ==========================

function isRealPageReload() {
    try {
        const [entry] = performance.getEntriesByType("navigation");
        if (entry) return entry.type === "reload";
    } catch (e) { /* ignore */ }

    // Fallback for older browsers
    if (performance.navigation) {
        return performance.navigation.type === performance.navigation.TYPE_RELOAD;
    }
    return false;
}

// Run this BEFORE restoring old messages below, and synchronously (not on
// the "load" event, which fires late) so a stale chat never flashes on
// screen for a split second after a real refresh.
if (isRealPageReload()) {
    sessionStorage.removeItem("chat");

    // Wipe old dataset / EDA / charts / chat context on the server too,
    // so any question asked after this point gets an honest
    // "no data loaded yet" answer instead of stale results.
    fetch("/session/reset", { method: "POST" }).catch(function (err) {
        console.warn("Session reset failed:", err);
    });
}

// ==========================
// Elements
// ==========================

const chatbot = document.getElementById("chatbot");
const toggle = document.getElementById("chat-toggle");
const sendBtn = document.getElementById("send-btn");
const input = document.getElementById("chat-input");
const body = document.getElementById("chat-body");

// ==========================
// Chat Memory
// ==========================

let messages = [];

const oldChat = sessionStorage.getItem("chat");

if (oldChat) {

    messages = JSON.parse(oldChat);

    messages.forEach(msg => {

        addMessage(msg.role, msg.text);

    });

}

// ==========================
// Open / Close Chat
// ==========================

toggle.addEventListener("click", function (e) {

    e.stopPropagation();

    chatbot.style.display =
        chatbot.style.display === "block"
            ? "none"
            : "block";

});

// Close when clicking outside

document.addEventListener("click", function (e) {

    if (
        !chatbot.contains(e.target) &&
        !toggle.contains(e.target)
    ) {

        chatbot.style.display = "none";

    }

});

// ==========================
// Send Button
// ==========================

sendBtn.addEventListener("click", sendMessage);

// ==========================
// Enter Key
// ==========================

input.addEventListener("keypress", function (e) {

    if (e.key === "Enter") {

        sendMessage();

    }

});

// ==========================
// Send Message
// ==========================

async function sendMessage() {

    const question = input.value.trim();

    if (question === "") return;

    addMessage("user", question);

    messages.push({

        role: "user",
        text: question

    });

    saveChat();

    input.value = "";

    // Loading

    const loading = document.createElement("div");

    loading.className = "bot";

    loading.id = "loading";

    loading.innerHTML = "Thinking...";

    body.appendChild(loading);

    scrollBottom();

    try {

        const response = await fetch("/chat", {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify({

                question: question

            })

        });

        const data = await response.json();

        loading.remove();

        addMessage("bot", data.answer);

        messages.push({

            role: "bot",
            text: data.answer

        });

        saveChat();

    }

    catch (err) {

        loading.remove();

        addMessage("bot", "❌ Connection Error");

        console.log(err);

    }

}

// ==========================
// Add Bubble
// ==========================

function addMessage(role, text) {

    const div = document.createElement("div");

    div.className = role;

    div.innerHTML = text;

    body.appendChild(div);

    scrollBottom();

}

// ==========================
// Scroll
// ==========================

function scrollBottom() {

    body.scrollTop = body.scrollHeight;

}

// ==========================
// Save Chat
// ==========================

function saveChat() {

    sessionStorage.setItem(

        "chat",

        JSON.stringify(messages)

    );

}

// ==========================
// Clear Chat
// ==========================

function clearChat() {

    messages = [];

    body.innerHTML = "";

    sessionStorage.removeItem("chat");

}
window.clearChat = clearChat;

// ==========================
// Suggestions
// ==========================

const suggestionButtons = document.querySelectorAll("#suggestions button");

suggestionButtons.forEach(btn => {

    btn.addEventListener("click", function () {

        input.value = this.textContent;

        sendMessage();

    });

});