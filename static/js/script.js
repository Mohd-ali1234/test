let isSidebarOpen = false;
const sidebar = document.getElementById('sidebar');
const mainContent = document.getElementById('main-content');
const menuIcon = document.getElementById('menu-icon');
const dashboardView = document.getElementById('dashboard-view');
const chatView = document.getElementById('chat-view');
const pillInfoView = document.getElementById('pill-info-view');
const reportAnalyzerView = document.getElementById('report-analyzer-view');
const inputBox = document.querySelector('input[placeholder="Type your next question or concern..."]');
const sendButton = document.querySelector('[id="sendBtn"]'); // The send button
const agentEnabled = document.getElementById("agentToggle").checked;
const chatMessagesContainer = document.getElementById('chat-messages');
const userId = localStorage.getItem("user_id"); // Replace dynamically if needed

const imageUpload = document.getElementById("imageUpload");
const imagePreviewContainer = document.getElementById("imagePreviewContainer");
const imagePreview = document.getElementById("imagePreview");
const imageError = document.getElementById("imageError");

const drugNameInput = document.getElementById("drugName");
const systemInstructionInput = document.getElementById("systemInstruction");

const analyzeButton = document.getElementById("analyzeButton");

const reportUpload = document.getElementById("reportUpload");
const reportTextInput = document.getElementById("reportText"); // optional context
const reportQueryInput = document.getElementById("reportQuery"); // optional query
const analyzeReportButton = document.getElementById("analyzeReportButton");
const reportLoadingIndicator = document.getElementById("reportLoadingIndicator");
const reportAnalysisText = document.getElementById("reportAnalysisText");
const reportErrorContainer = document.getElementById("reportErrorContainer");
const reportErrorMessage = document.getElementById("reportErrorMessage");


const loadingIndicator = document.getElementById("loadingIndicator");
const resultsContainer = document.getElementById("resultsContainer");
const analysisText = document.getElementById("analysisText");
const errorContainer = document.getElementById("errorContainer");
const errorMessage = document.getElementById("errorMessage");

if (!userId) {
    // No user is logged in — redirect to login
    window.location.replace("/login");
}
function checkInputs() {
    analyzeButton.disabled = !(
        drugNameInput.value.trim() || imageUpload.files.length
    );
}

reportUpload.addEventListener("change", () => {
    analyzeReportButton.disabled = reportUpload.files.length === 0;

    const file = reportUpload.files[0];
    const previewContainer = document.getElementById("reportPreviewContainer");
    previewContainer.classList.remove("hidden");
    previewContainer.innerHTML = ""; // Clear previous preview

    if (!file) return;

    if (file.type === "application/pdf") {
        // Show PDF icon + filename
        const pdfPreview = document.createElement("div");
        pdfPreview.className = "flex flex-col items-center justify-center p-4 text-gray-600 bg-gray-100 rounded-lg";
        pdfPreview.innerHTML = `
            <svg class="w-12 h-12 mb-2 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M12 2l8 8-8 8-8-8 8-8zm0 0v18" />
            </svg>
            <p class="text-sm font-medium">${file.name}</p>
            <p class="text-xs text-gray-500">${(file.size / 1024 / 1024).toFixed(2)} MB</p>
        `;
        previewContainer.appendChild(pdfPreview);
    } else if (file.type.startsWith("image/")) {
        // Show image preview
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.className = "max-w-full h-auto max-h-48 mx-auto rounded-lg shadow-md";
        img.alt = "Report Preview";
        previewContainer.appendChild(img);
    } else {
        previewContainer.innerHTML = `<p class="text-sm text-red-500">Unsupported file type</p>`;
    }
});


// --- Sidebar & Mobile Toggling ---
const toggleSidebar = () => {
    isSidebarOpen = !isSidebarOpen;
    if (isSidebarOpen) {
        sidebar.classList.remove('sidebar-hidden');
        menuIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>'; // Close icon (X)
    } else {
        sidebar.classList.add('sidebar-hidden');
        menuIcon.innerHTML = '<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>'; // Hamburger icon
    }
};

// Initialize sidebar state on small screens
if (window.innerWidth < 768) {
    sidebar.classList.add('sidebar-hidden');
}

// --- View Switching Logic ---
const showView = (viewName, activeElement) => {
    // Update Navigation Menu Active State
    document.querySelectorAll('.menu-link').forEach(link => {
        link.classList.remove('active');
    });
    if (activeElement) {
        activeElement.classList.add('active');
    }

    // Toggle Views
    if (viewName === 'dashboard') {
        dashboardView.classList.remove('hidden');
        chatView.classList.add('hidden');
        pillInfoView.classList.add('hidden');
        reportAnalyzerView.classList.add('hidden');
        // Ensure chat interface scrolling is reset
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
    } else if (viewName === 'chat') {
        dashboardView.classList.add('hidden');
        chatView.classList.remove('hidden');
        pillInfoView.classList.add('hidden');
        reportAnalyzerView.classList.add('hidden');
        simulateAITyping();
    } else if (viewName === 'pill') {
        dashboardView.classList.add('hidden');
        chatView.classList.add('hidden');
        pillInfoView.classList.remove('hidden');
        reportAnalyzerView.classList.add('hidden');
    }else if (viewName === 'report') {
        dashboardView.classList.add('hidden');
        chatView.classList.add('hidden');
        pillInfoView.classList.add('hidden');
        reportAnalyzerView.classList.remove('hidden');
    }

    // Close sidebar on mobile after clicking a link
    if (window.innerWidth < 768 && isSidebarOpen) {
        toggleSidebar();
    }
};

// --- AI Chat Typing Simulation ---
const showLastAIResponseWithTyping = () => {
    const lastChat = chatMessagesContainer.lastChild;
    if (!lastChat) return;

    const aiTextElement = lastChat.querySelector('p.text-sm');
    const fullResponse = aiTextElement.innerText;
    aiTextElement.innerText = "";

    let i = 0;
    const timer = setInterval(() => {
        if (i < fullResponse.length) {
            aiTextElement.innerHTML += fullResponse[i] === "\n" ? "<br>" : fullResponse[i];
            chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
            i++;
        } else {
            clearInterval(timer);
        }
    }, 20);
};

const showAIWelcomeMessage = () => {
    const chatMessagesContainer = document.getElementById('chat-messages');

    const welcomeDiv = document.createElement("div");
    welcomeDiv.className = "flex justify-start";
    welcomeDiv.innerHTML = `
        <div class="max-w-3xl bg-soft-blue text-gray-800 p-4 rounded-t-xl rounded-br-xl shadow-md">
            <p class="font-medium text-blue-800 mb-2">MediAssist AI (v2.0)</p>
            <p class="text-sm">
                Hello! Please describe your symptoms in detail, or upload any relevant reports. I am here to provide initial, non-diagnostic guidance.
            </p>
            <div class="mt-2 text-xs text-blue-500 bg-blue-100 p-1 rounded-lg">
                <span class="font-semibold">Disclaimer:</span> I am an AI and cannot replace a doctor. Always consult a licensed medical professional for diagnosis or treatment.
            </div>
        </div>
    `;

    chatMessagesContainer.appendChild(welcomeDiv);
};

const fetchChatHistory = async () => {
    try {
        const response = await fetch(`https://test-2-4mru.onrender.com/chats/${userId}`);
        const data = await response.json();

        // Clear existing messages
        chatMessagesContainer.innerHTML = "";

        // If no chat history → show welcome message
        if (!data.chats || data.chats.length === 0) {
            showAIWelcomeMessage();
            return; // stop execution
        }

        // Render messages
        data.chats.forEach(chat => {
            if (chat.user_message) {
                const userDiv = document.createElement("div");
                userDiv.className = "flex justify-end";
                userDiv.innerHTML = `
                    <div class="max-w-3xl bg-blue-600 text-white p-4 rounded-t-xl rounded-bl-xl shadow-md">
                        <p class="text-sm">${chat.user_message}</p>
                    </div>
                `;
                chatMessagesContainer.appendChild(userDiv);
            }

            if (chat.ai_response) {
                const aiDiv = document.createElement("div");
                aiDiv.className = "flex justify-start";

                const formattedResponse = chat.ai_response
                    .replace(/\n/g, "<br>")
                    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

                aiDiv.innerHTML = `
                    <div class="max-w-3xl bg-soft-blue text-gray-800 p-4 rounded-t-xl rounded-br-xl shadow-md">
                        <p class="font-medium text-blue-800 mb-2">MediAssist AI (v2.0)</p>
                        <p class="text-sm">${formattedResponse}</p>
                    </div>
                `;

                chatMessagesContainer.appendChild(aiDiv);
            }
        });
        // Scroll to bottom
        chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;

    } catch (error) {
        console.error("Error fetching chat history:", error);
    }
};



const sendChat = async () => {
    const userMessage = inputBox.value.trim();
    if (!userMessage) return;

    const agentEnabled = document.getElementById("agentToggle").checked;

    // 1️⃣ Append user message
    const userDiv = document.createElement("div");
    userDiv.className = "flex justify-end";
    userDiv.innerHTML = `<div class="max-w-3xl bg-blue-600 text-white p-4 rounded-t-xl rounded-bl-xl shadow-md">
                            <p class="text-sm">${userMessage}</p>
                         </div>`;
    chatMessagesContainer.appendChild(userDiv);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
    inputBox.value = "";

    // 2️⃣ Append loader for AI response
    const aiDiv = document.createElement("div");
    aiDiv.className = "flex justify-start";
    aiDiv.innerHTML = `
        <div class="max-w-3xl bg-soft-blue text-gray-800 p-4 rounded-t-xl rounded-br-xl shadow-md">
            <p class="font-medium text-blue-800 mb-2">MediAssist AI (v2.0)</p>
            <p class="text-sm loader">Typing<span class="dots">...</span></p>
        </div>`;
    chatMessagesContainer.appendChild(aiDiv);
    chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;

    // Animate loader dots
    const dots = aiDiv.querySelector(".dots");
    let dotCount = 0;
    const loaderInterval = setInterval(() => {
        dotCount = (dotCount + 1) % 4;
        dots.textContent = ".".repeat(dotCount);
    }, 500);

    // 3️⃣ Call API
    let aiResponse = "";
    try {
        const res = await fetch("https://test-2-4mru.onrender.com/chat/send", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                user_id: userId,
                user_message: userMessage,
                mode: agentEnabled ? "agent" : "normal"
            })
        });

        const data = await res.json();
        if (data.success && data.chat.ai_response) {
            aiResponse = data.chat.ai_response;
        } else {
            aiResponse = "Sorry, I couldn't get a response. Please try again.";
        }
    } catch (error) {
        console.error("Error calling send API:", error);
        aiResponse = "Sorry, something went wrong. Please try again.";
    }

    // 4️⃣ Stop loader
    clearInterval(loaderInterval);

    // 5️⃣ Typewriter effect with HTML handling
    const formattedResponse = aiResponse
        .replace(/\n/g, "<br>")
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

    const responseParagraph = aiDiv.querySelector("p.loader");
    responseParagraph.innerHTML = ""; // Clear loader

    function typeWriterHTML(element, html, speed = 20) {
        let i = 0;
        let tag = false;
        let tagBuffer = "";

        const interval = setInterval(() => {
            if (i < html.length) {
                const char = html[i];

                if (char === "<") tag = true;

                if (tag) tagBuffer += char;
                else element.innerHTML += char;

                if (char === ">") {
                    element.innerHTML += tagBuffer;
                    tagBuffer = "";
                    tag = false;
                }

                i++;
                chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
            } else {
                clearInterval(interval);
            }
        }, speed);
    }

    // Start typing animation
    typeWriterHTML(responseParagraph, formattedResponse, 15); // 15ms per character


}

// Event listeners
sendButton.addEventListener("click", sendChat);
inputBox.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
});

// --- Profile Dropdown Toggling ---
const dropdown = document.getElementById('profile-dropdown');
const toggleDropdown = () => {
    dropdown.classList.toggle('hidden');
};

// Close dropdown when clicking outside
document.addEventListener('click', (event) => {
    const profileArea = document.getElementById('profile-area');
    if (profileArea && !profileArea.contains(event.target)) {
        dropdown.classList.add('hidden');
    }
});

drugNameInput.addEventListener("input", checkInputs);
imageUpload.addEventListener("change", () => {
    const file = imageUpload.files[0];
    if (file) {
        // Preview image
        const reader = new FileReader();
        reader.onload = e => {
            imagePreview.src = e.target.result;
            imagePreviewContainer.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    } else {
        imagePreviewContainer.classList.add("hidden");
    }
    checkInputs();
});

analyzeReportButton.addEventListener("click", async () => {
    // Scroll to results and reset previous states
    reportAnalysisText.scrollIntoView({ behavior: "smooth" });
    reportLoadingIndicator.classList.remove("hidden");
    reportAnalysisText.innerHTML = "";
    reportErrorContainer.classList.add("hidden");
    reportErrorMessage.textContent = "";

    try {
        const formData = new FormData();

        // Append report file
        if (reportUpload.files.length > 0) {
            formData.append("report_file", reportUpload.files[0]);
        } else {
            throw new Error("Please upload a report file.");
        }

        // Optional: append custom analysis focus or query
        if (reportQueryInput.value.trim()) {
            formData.append("custom_analysis_focus", reportQueryInput.value.trim());
        }

        if (reportTextInput.value.trim()) {
            formData.append("custom_analysis_focus", reportTextInput.value.trim());
        }
        

        const response = await fetch("https://test-2-4mru.onrender.com/analyze-report", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        reportLoadingIndicator.classList.add("hidden");

        if (response.ok) {
            // Display the analysis result
            const formattedHtml = parseMarkdown(data.analysis);
            reportAnalysisText.innerHTML = formattedHtml;
            // reportErrorContainer.classList.remove("hidden");
            // reportErrorMessage.textContent = data.error || "Something went wrong.";
        }

    } catch (err) {
        reportLoadingIndicator.classList.add("hidden");
        reportErrorContainer.classList.remove("hidden");
        reportErrorMessage.textContent = err.message || "Network error";
    }
});

// Handle Analyze Button Click
analyzeButton.addEventListener("click", async () => {
    // Hide previous results/errors
    resultsContainer.scrollIntoView({ behavior: "smooth" });
    loadingIndicator.classList.remove("hidden");
    analysisText.innerHTML = "";
    errorContainer.classList.add("hidden");
    errorMessage.textContent = "";

    try {
        const formData = new FormData();

        // Append drug name if exists
        if (drugNameInput.value.trim()) {
            formData.append("drug_name", drugNameInput.value.trim());
        }

        // Append image if exists
        if (imageUpload.files.length > 0) {
            formData.append("drug_image", imageUpload.files[0]);
        }

        // Optional: append custom instructions if needed
        if (systemInstructionInput.value.trim()) {
            formData.append("custom_analysis_focus", systemInstructionInput.value.trim());
        }

        const response = await fetch("https://test-2-4mru.onrender.com/analyze-drug", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        loadingIndicator.classList.add("hidden");

        if (response.ok) {
            const formattedHtml = parseMarkdown(data.analysis);
            analysisText.innerHTML = formattedHtml;
        } else {
            errorContainer.classList.remove("hidden");
            errorMessage.textContent = data.error || "Something went wrong.";
        }

    } catch (err) {
        loadingIndicator.classList.add("hidden");
        errorContainer.classList.remove("hidden");
        errorMessage.textContent = err.message || "Network error";
    }
});

function parseMarkdown(mdText) {
    if (!mdText) return "";

    let html = mdText;

    // Headings
    html = html.replace(/^### (.*)$/gm, '<h3 class="text-xl font-bold mt-4 mb-2">$1</h3>');
    html = html.replace(/^## (.*)$/gm, '<h2 class="text-2xl font-bold mt-4 mb-2">$1</h2>');
    html = html.replace(/^# (.*)$/gm, '<h1 class="text-3xl font-bold mt-4 mb-2">$1</h1>');

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

    // Lists
    html = html.replace(/^\* (.*)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');

    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

    return html;
}


window.onload = () => {
    showView('dashboard', document.getElementById('nav-home'));
    fetchChatHistory(); // Load chat history when page loads
    showAIWelcomeMessage();  // Add welcome message at the top
};


