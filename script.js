const table = document.getElementById("table");

function detectFace() {
    fetch("/detect")
        .then(res => res.json())
        .then(data => {

            table.innerHTML = `
            <tr>
                <th>Name</th>
                <th>Date</th>
                <th>Time</th>
            </tr>
            `;

            if (data.attendance) {
                table.innerHTML += `
                <tr>
                    <td>${data.attendance.name}</td>
                    <td>${data.attendance.date}</td>
                    <td>${data.attendance.time}</td>
                </tr>
                `;
            }

            if (!data.stop) {
                setTimeout(detectFace, 500);
            }
        });
}

detectFace();
