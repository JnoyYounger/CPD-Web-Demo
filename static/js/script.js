document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM 加载完成');
    const uploadButton = document.getElementById('uploadButton');
    const originalImg = document.getElementById('originalImage');
    const foregroundImg = document.getElementById('foregroundImage');
    console.log('初始 DOM 检查：', { uploadButton, originalImg, foregroundImg });

    if (!uploadButton) {
        console.error('未找到上传按钮');
        alert('页面加载错误：未找到上传按钮');
        return;
    }
    if (!originalImg || !foregroundImg) {
        console.error('未找到图像元素');
        alert('页面加载错误：找不到 originalImage 或 foregroundImage');
        return;
    }

    uploadButton.addEventListener('click', uploadImage);

    function uploadImage() {
        const imageInput = document.getElementById('imageInput');
        const gtInput = document.getElementById('gtInput');
        const metricsDiv = document.getElementById('metrics');
        const maeTd = document.getElementById('mae');
        const fmeasureTd = document.getElementById('fmeasure');
        const precisionTd = document.getElementById('precision');
        const recallTd = document.getElementById('recall');
        const iouTd = document.getElementById('iou');

        console.log('上传时 DOM 检查：', { imageInput, gtInput, originalImg, foregroundImg });
        if (!imageInput.files.length) {
            alert('请选择一张图像');
            return;
        }

        const formData = new FormData();
        formData.append('file', imageInput.files[0]);
        if (gtInput.files.length) {
            formData.append('gt_file', gtInput.files[0]);
        }

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`服务器错误：${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('后端响应：', data);
            if (!data.original || !data.foreground) {
                throw new Error('后端未返回完整图像路径');
            }
            originalImg.src = data.original + '?t=' + new Date().getTime();
            originalImg.style.display = 'block';
            foregroundImg.src = data.foreground + '?t=' + new Date().getTime();
            foregroundImg.style.display = 'block';

            if (data.metrics && Object.keys(data.metrics).length) {
                maeTd.textContent = data.metrics.mae.toFixed(4);
                fmeasureTd.textContent = data.metrics.fmeasure.toFixed(4);
                precisionTd.textContent = data.metrics.precision.toFixed(4);
                recallTd.textContent = data.metrics.recall.toFixed(4);
                iouTd.textContent = data.metrics.iou.toFixed(4);
                metricsDiv.style.display = 'block';
            } else {
                metricsDiv.style.display = 'none';
            }

            imageInput.value = '';
            gtInput.value = '';
        })
        .catch(error => {
            console.error('错误:', error);
            alert('处理图像失败：' + error.message);
        });
    }
});