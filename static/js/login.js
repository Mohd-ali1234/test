// ===============================
//  COMMON VALIDATION FUNCTIONS
// ===============================
function validateEmail(email) {
    const regex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return regex.test(email);
}

function validatePassword(password) {
    return password.length >= 6; // Minimum 6 chars
}

function showError(message) {
    alert(message); // you can replace with UI popup later
}

// ===============================
//  REGISTER FORM HANDLER
// ===============================
const registerForm = document.querySelector(".register-form h2");
let isRegister = false;

if (registerForm && registerForm.textContent.includes("Create Your Account")) {
    isRegister = true;
}

const form = document.querySelector(".register-form");

if (form) {
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const email = document.getElementById("email").value.trim();
        const password = document.getElementById("password").value.trim();

        // ===============================
        //  VALIDATIONS
        // ===============================
        if (!email || !password) {
            showError("All fields are required!");
            return;
        }

        if (!validateEmail(email)) {
            showError("Please enter a valid email address.");
            return;
        }

        if (!validatePassword(password)) {
            showError("Password must be at least 6 characters long.");
            return;
        }

        // ===============================
        //  API CALL
        // ===============================
        const endpoint = isRegister
            ? "http://localhost:8000/auth/register"
            : "http://localhost:8000/auth/login";

        const formData = new FormData();
        formData.append("email", email);
        formData.append("password", password);

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData,
            });
        
            const data = await response.json();
        
            // 🔥 Print full response
            console.log("API Response:", data);
        
            if (!response.ok) {
                showError(data.detail || "Something went wrong");
                return;
            }
        
            if (data.user_id) {
                localStorage.setItem("user_id", data.user_id);
            }
        
            if (isRegister) {
                window.location.replace("/"); // or "../index.html" depending on path
            } else {
                window.location.replace("/"); // or "../index.html" depending on path
            }
        
        } catch (err) {
            console.error(err);
            showError("Server error. Please try again later.");
        }
        
    });
}
