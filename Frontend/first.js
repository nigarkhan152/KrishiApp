async function uploadImage() {
    let file = document.getElementById("fileInput").files[0];
    let formData = new FormData();
    formData.append("file", file);

    let response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    let data = await response.json();
    document.getElementById("result").innerText = "Prediction: " + data.prediction;
}
