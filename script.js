let interval = null;

function startCamera() {
    fetch("/start_camera", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            if (data.status === "camera started") {
                document.getElementById('video').src = "/video_feed";
                if(interval) clearInterval(interval);
                interval = setInterval(getLastAttendance, 1000);
            } else {
                alert(data.status);
            }
        });
}

function stopCamera() {
    fetch("/stop_camera", { method: "POST" })
        .then(res => res.json())
        .then(data => {
            if(interval) clearInterval(interval);
            document.getElementById('video').src = "";
        });
}

function getLastAttendance() {
    fetch("/get_last_attendance")
        .then(res => res.json())
        .then(data => {
            const tbody = document.querySelector("#attendance-table tbody");
            tbody.innerHTML = "";
            if(data) {
                const tr = document.createElement("tr");
                tr.innerHTML = `<td>${data.name}</td><td>${data.date}</td><td>${data.time}</td>`;
                tbody.appendChild(tr);
            }
        });
}
