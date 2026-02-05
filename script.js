function checkStatus() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            if (data.name) {
                document.getElementById("videoFeed").src = "";
                document.getElementById("cameraSection").style.display = "none";
                document.getElementById("successBox").style.display = "block";

                document.getElementById("details").innerHTML =
                    "Name: " + data.name +
                    "<br>Date: " + data.date +
                    "<br>Time: " + data.time;
            }
        });
}

setInterval(checkStatus, 1000);
