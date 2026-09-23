// ==========================================
// CANVAS
// ==========================================

const canvas =
    document.getElementById("drawingCanvas");


const ctx =
    canvas.getContext("2d");


// Drawing settings

ctx.lineWidth = 12;

ctx.lineCap = "round";

ctx.lineJoin = "round";

ctx.strokeStyle = "black";


// Is the user currently drawing?

let drawing = false;


// ==========================================
// START DRAWING
// ==========================================

canvas.addEventListener(
    "mousedown",
    function(event) {

        drawing = true;

        ctx.beginPath();

        ctx.moveTo(
            event.offsetX,
            event.offsetY
        );

    }
);


// ==========================================
// DRAW
// ==========================================

canvas.addEventListener(
    "mousemove",
    function(event) {

        if (!drawing) {
            return;
        }


        ctx.lineTo(
            event.offsetX,
            event.offsetY
        );


        ctx.stroke();

    }
);


// ==========================================
// STOP DRAWING
// ==========================================

window.addEventListener(
    "mouseup",
    function() {

        drawing = false;

    }
);


// ==========================================
// CLEAR BUTTON
// ==========================================

const clearButton =
    document.getElementById(
        "clearButton"
    );


clearButton.addEventListener(
    "click",
    function() {

        ctx.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );


        document.getElementById(
            "prediction"
        ).textContent =
            "Prediction: —";


        document.getElementById(
            "confidence"
        ).textContent = "";

    }
);


// ==========================================
// RECOGNISE BUTTON
// ==========================================

const predictButton =
    document.getElementById(
        "predictButton"
    );


predictButton.addEventListener(
    "click",
    async function() {

        // Convert canvas to PNG

        const imageData =
            canvas.toDataURL(
                "image/png"
            );


        // Change text while waiting

        document.getElementById(
            "prediction"
        ).textContent =
            "Recognising...";


        try {

            // Send image to Flask

            const response = await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        image: imageData
                    })
                }
            );


            const data =
                await response.json();


            // Check for Flask errors

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "Prediction failed"
                );

            }


            // Display prediction

            document.getElementById(
                "prediction"
            ).textContent =
                "Prediction: " +
                data.prediction;


            // Display confidence if available

            if (
                data.confidence !==
                undefined
            ) {

                document.getElementById(
                    "confidence"
                ).textContent =
                    "Confidence: " +
                    data.confidence +
                    "%";

            }

        }

        catch (error) {

            console.error(error);


            document.getElementById(
                "prediction"
            ).textContent =
                "Something went wrong.";

        }

    }
);