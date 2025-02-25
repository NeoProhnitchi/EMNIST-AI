const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

// Canvas setup
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 15;
ctx.lineCap = 'round';

// Drawing functions
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
}

function draw(e) {
    if (!isDrawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('prediction').textContent = '';
}

async function predict() {
    // Preprocess image
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const processed = preprocessImage(imageData);

    // Send to backend
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: processed })
    });

    const prediction = await response.json();
    document.getElementById('prediction').textContent = prediction.character;
}

function preprocessImage(imageData) {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    
    // Resize to 28x28
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    // Convert to grayscale and normalize
    const data = tempCtx.getImageData(0, 0, 28, 28).data;
    const grayscale = [];
    
    for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] + data[i+1] + data[i+2]) / 3;
        grayscale.push(255 - avg);  // Invert colors (EMNIST style)
    }
    
    return grayscale.map(x => x / 255);  // Normalize
}