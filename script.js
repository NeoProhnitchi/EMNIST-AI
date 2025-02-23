const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const colorPicker = document.getElementById('colorPicker');
const lineWidth = document.getElementById('lineWidth');
const clearBtn = document.getElementById('clearBtn');

// Set canvas size
canvas.width = 800;
canvas.height = 500;

// Drawing state
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize canvas
ctx.strokeStyle = colorPicker.value;
ctx.lineWidth = lineWidth.value;
ctx.lineCap = 'round';

// Event Listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

colorPicker.addEventListener('input', updateColor);
lineWidth.addEventListener('input', updateLineWidth);
clearBtn.addEventListener('click', clearCanvas);

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing) return;
    
    ctx.strokeStyle = colorPicker.value;
    ctx.lineWidth = lineWidth.value;
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function stopDrawing() {
    isDrawing = false;
}

function updateColor() {
    ctx.strokeStyle = this.value;
}

function updateLineWidth() {
    ctx.lineWidth = this.value;
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}