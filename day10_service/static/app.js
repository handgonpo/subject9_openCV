const sampleSelect =
    document.getElementById(
        "sampleSelect"
    );

const inspectButton =
    document.getElementById(
        "inspectButton"
    );

const previewImage =
    document.getElementById(
        "previewImage"
    );

const finalResult =
    document.getElementById(
        "finalResult"
    );


async function loadSamples() {

    const response =
        await fetch(
            "/api/samples"
        );

    const data =
        await response.json();


    for (
        const fileName
        of data.samples
    ) {
        const option =
            document.createElement(
                "option"
            );

        option.value =
            fileName;

        option.textContent =
            fileName;

        sampleSelect.appendChild(
            option
        );
    }
}


sampleSelect.addEventListener(
    "change",
    () => {

        const fileName =
            sampleSelect.value;

        if (!fileName) {

            previewImage.removeAttribute(
                "src"
            );

            previewImage.hidden = true;

            return;
        }

        previewImage.src =
            `/api/sample-image/${encodeURIComponent(fileName)}`;

        previewImage.hidden = false;
    }
);


inspectButton.addEventListener(
    "click",
    async () => {

        const fileName =
            sampleSelect.value;

        if (!fileName) {
            alert(
                "검사할 제품을 선택하세요."
            );

            return;
        }


        finalResult.textContent =
            "INSPECTING...";

        finalResult.className =
            "final waiting";


        const response =
            await fetch(
                "/api/inspect",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json",
                    },

                    body:
                        JSON.stringify(
                            {
                                file_name:
                                    fileName,
                            }
                        ),
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            finalResult.textContent =
                "ERROR";

            finalResult.className =
                "final ng";

            alert(
                data.detail
                || "검사 중 오류가 발생했습니다."
            );

            return;
        }


        renderResult(
            data
        );

        await loadStats();
        await loadHistory();
    }
);


function renderResult(
    data
) {

    finalResult.textContent =
        data.final_decision;

    finalResult.className =
        data.final_decision
        === "OK"
        ? "final ok"
        : "final ng";


    document.getElementById(
        "ocrText"
    ).textContent =
        data.measurements.ocr_text
        || "(empty)";


    document.getElementById(
        "expectedText"
    ).textContent =
        data.measurements.expected_text;


    document.getElementById(
        "reasonText"
    ).textContent =
        data.reasons.length
        ? data.reasons.join(", ")
        : "PASS";


    document.getElementById(
        "ruleVersion"
    ).textContent =
        data.rule_version;


    document.getElementById(
        "serviceVersion"
    ).textContent =
        data.service_version;


    document.getElementById(
        "requestId"
    ).textContent =
        data.request_id;
}


async function loadStats() {

    const response =
        await fetch(
            "/api/stats"
        );

    const data =
        await response.json();


    document.getElementById(
        "totalCount"
    ).textContent =
        data.total;

    document.getElementById(
        "okCount"
    ).textContent =
        data.ok;

    document.getElementById(
        "ngCount"
    ).textContent =
        data.ng;
}


async function loadHistory() {

    const response =
        await fetch(
            "/api/history?limit=10"
        );

    const data =
        await response.json();

    const body =
        document.getElementById(
            "historyBody"
        );

    body.innerHTML = "";


    const rows =
        [...data.history]
        .reverse();


    for (
        const item
        of rows
    ) {

        const tr =
            document.createElement(
                "tr"
            );

        const reason =
            item.reasons.length
            ? item.reasons.join(", ")
            : "PASS";


        tr.innerHTML = `
            <td>${item.inspected_at}</td>
            <td>${item.file_name}</td>
            <td>${item.final_decision}</td>
            <td>${reason}</td>
        `;

        body.appendChild(
            tr
        );
    }
}


async function init() {
    await loadSamples();
    await loadStats();
    await loadHistory();
}


init();