document.addEventListener('DOMContentLoaded', () => {
    // Hamburger menu toggle
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');

    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('active');
    });

    // Contact form submission
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(contactForm);
            const response = await fetch('/contact', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            showPopup(result.message);
            contactForm.reset();
        });
    }

    // Symptom form submission
    const symptomForm = document.getElementById('symptom-form');
    if (symptomForm) {
        symptomForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(symptomForm);
            const response = await fetch('/symptom', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            let message = `<strong>Disease:</strong> ${result.disease}<br>`;
            message += `<strong>Description:</strong> ${result.description}<br>`;
            message += `<strong>Precautions:</strong><ul>`;
            result.precautions.forEach(p => message += `<li>${p}</li>`);
            message += `</ul>`;
            showPopup(message);
            symptomForm.reset();
        });
    }

    // Heart form submission
    const heartForm = document.getElementById('heart-form');
    if (heartForm) {
        heartForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(heartForm);
            const response = await fetch('/heart', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            showPopup(result.result);
            heartForm.reset();
        });
    }

    // Lung form submission
    const lungForm = document.getElementById('lung-form');
    if (lungForm) {
        lungForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(lungForm);
            const response = await fetch('/lung', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            showPopup(result.result);
            lungForm.reset();
        });
    }

    // Thyroid form submission
    const thyroidForm = document.getElementById('thyroid-form');
    if (thyroidForm) {
        thyroidForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(thyroidForm);
            const response = await fetch('/thyroid', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            showPopup(result.result);
            thyroidForm.reset();
        });
    }

    // Diabetes form submission
    const diabetesForm = document.getElementById('diabetes-form');
    if (diabetesForm) {
        diabetesForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(diabetesForm);
            const response = await fetch('/diabetes', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            showPopup(result.result);
            diabetesForm.reset();
        });
    }
});

function showPopup(message) {
    const popup = document.getElementById('popup');
    const popupMessage = document.getElementById('popup-message');
    popupMessage.innerHTML = message;
    popup.style.display = 'flex';
}

function closePopup() {
    const popup = document.getElementById('popup');
    popup.style.display = 'none';
}