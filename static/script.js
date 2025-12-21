document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("file-input");
    const submitButton = document.getElementById("submit-button");
    const statusDiv = document.getElementById("status");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        if (!fileInput.files || fileInput.files.length === 0) {
            statusDiv.textContent = "Please select a file first.";
            statusDiv.style.color = "red";
            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append("file", file);
        submitButton.disabled = true;
        statusDiv.textContent = "Uploading and processing... This may take several minutes.";
        statusDiv.style.color = "#0056b3";

        try {
            const response = await fetch("/predict/", {
                method: "POST",
                body: formData,
            });

            if (response.ok) {
                const disposition = response.headers.get("Content-Disposition");
                let filename = "generated_ct.nii.gz";
                if (disposition && disposition.indexOf("attachment") !== -1) {
                    const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
                    const matches = filenameRegex.exec(disposition);
                    if (matches != null && matches[1]) {
                        filename = matches[1].replace(/['"]/g, "");
                    }
                }

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                
                const a = document.createElement("a");
                a.style.display = "none";
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                
                window.URL.revokeObjectURL(url);
                a.remove();
                
                statusDiv.textContent = "Success! Your CT scan is downloading.";
                statusDiv.style.color = "green";

            } else {
                const errorData = await response.json();
                statusDiv.textContent = `Error: ${errorData.detail || response.statusText}`;
                statusDiv.style.color = "red";
            }

        } catch (error) {
            console.error("Fetch error:", error);
            statusDiv.textContent = "An error occurred. Check the server logs.";
            statusDiv.style.color = "red";
        } finally {
            
            submitButton.disabled = false;
        }
    });
});