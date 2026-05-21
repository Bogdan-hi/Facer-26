const fileInput = document.getElementById("fileInput");

const resultDiv = document.getElementById("result");

const preview = document.getElementById("preview");

const cameraBtn = document.getElementById("cameraBtn");

// =====================================================
// SHOW RESULT
// =====================================================

function showResult(success, message) {

    resultDiv.style.display = "block";

    resultDiv.innerHTML = message;

    if (success) {

        resultDiv.className = "success";

    } else {

        resultDiv.className = "error";
    }
}

// =====================================================
// UPLOAD IMAGE
// =====================================================

fileInput.addEventListener("change", async () => {

    const file = fileInput.files[0];

    if (!file) return;

    // preview

    preview.src = URL.createObjectURL(file);

    preview.style.display = "block";

    const formData = new FormData();

    formData.append("file", file);

    const response = await fetch("/upload", {

        method: "POST",
        body: formData
    });

    const data = await response.json();

    showResult(
        data.success,
        data.message
    );
});

// =====================================================
// CAMERA CAPTURE
// =====================================================

cameraBtn.addEventListener("click", async () => {

    showResult(false, "Запуск камеры...");

    const response = await fetch("/capture", {
        method: "POST"
    });

    const data = await response.json();

    if (data.image) {

        preview.src = data.image;

        preview.style.display = "block";
    }

    showResult(
        data.success,
        data.message
    );
});