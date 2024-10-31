const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileInput');
const genreOutput = document.getElementById('genreOutput');

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
});

dropArea.addEventListener('dragover', () => {
    dropArea.classList.add('dragover');
});

dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('dragover');
});

dropArea.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    handleFiles(files);
});

fileInput.addEventListener('change', () => {
    handleFiles(fileInput.files);
});

function handleFiles(files) {
    const file = files[0];
    if (file && file.type.startsWith('audio/')) {
        genreOutput.textContent = "Processing...";
        
        // Placeholder: Replace with actual API call
        setTimeout(() => {
            genreOutput.textContent = "Genre: Rock (sample result)";
        }, 2000);
    } else {
        genreOutput.textContent = "Please upload an audio file.";
    }
}
