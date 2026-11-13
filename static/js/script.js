// script.js
// Simple client-side validation for the prediction form.
// Server-side (Flask) validation is still the source of truth;
// this just gives the user instant feedback.

document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("predict-form");
  if (!form) return;

  form.addEventListener("submit", function (event) {
    let isValid = true;
    const errorMessages = [];

    const requiredFields = form.querySelectorAll("[required]");
    requiredFields.forEach(function (field) {
      clearFieldError(field);

      const value = field.value.trim();

      if (value === "") {
        isValid = false;
        showFieldError(field, "This field is required.");
        errorMessages.push(`${field.name} is required.`);
        return;
      }

      if (field.type === "number") {
        const numValue = Number(value);
        if (Number.isNaN(numValue)) {
          isValid = false;
          showFieldError(field, "Please enter a valid number.");
          errorMessages.push(`${field.name} must be a valid number.`);
          return;
        }
        const min = field.getAttribute("min");
        const max = field.getAttribute("max");
        if (min !== null && numValue < Number(min)) {
          isValid = false;
          showFieldError(field, `Value must be at least ${min}.`);
        }
        if (max !== null && numValue > Number(max)) {
          isValid = false;
          showFieldError(field, `Value must be at most ${max}.`);
        }
      }
    });

    if (!isValid) {
      event.preventDefault();
    }
  });

  function showFieldError(field, message) {
    field.style.borderColor = "#f87171";
    let errorEl = field.parentElement.querySelector(".js-error");
    if (!errorEl) {
      errorEl = document.createElement("div");
      errorEl.className = "js-error";
      errorEl.style.color = "#b91c1c";
      errorEl.style.fontSize = "0.78rem";
      errorEl.style.marginTop = "2px";
      field.parentElement.appendChild(errorEl);
    }
    errorEl.textContent = message;
  }

  function clearFieldError(field) {
    field.style.borderColor = "";
    const errorEl = field.parentElement.querySelector(".js-error");
    if (errorEl) {
      errorEl.remove();
    }
  }
});
