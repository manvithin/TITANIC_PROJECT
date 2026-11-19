// Small enhancements for the Titanic Survival Lab website.

const SAMPLES = {
  "first-class": {
    pclass: "1",
    sex: "female",
    age: "29",
    sibsp: "1",
    parch: "0",
    fare: "211.34",
    embarked: "S",
  },
  "third-class": {
    pclass: "3",
    sex: "male",
    age: "25",
    sibsp: "0",
    parch: "0",
    fare: "7.25",
    embarked: "S",
  },
  family: {
    pclass: "2",
    sex: "female",
    age: "35",
    sibsp: "1",
    parch: "2",
    fare: "26.0",
    embarked: "C",
  },
};

const form = document.getElementById("predict-form");

if (form) {
  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => {
      const sample = SAMPLES[button.dataset.sample];
      if (!sample) return;

      Object.entries(sample).forEach(([name, value]) => {
        const field = form.elements.namedItem(name);
        if (field) field.value = value;
      });

      button.blur();
      form.querySelector('button[type="submit"]').focus({ preventScroll: true });
    });
  });

  // Avoid double submissions while the server responds.
  form.addEventListener("submit", () => {
    const submit = form.querySelector('button[type="submit"]');
    if (submit) {
      submit.disabled = true;
      submit.textContent = "Predicting...";
    }
  });
}
