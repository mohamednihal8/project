document.addEventListener('DOMContentLoaded', function() {
    const inputs = document.querySelectorAll('.tooltip-input');
    inputs.forEach(input => {
        input.addEventListener('mouseenter', function() {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.innerText = input.dataset.tooltip;
            document.body.appendChild(tooltip);
            const rect = input.getBoundingClientRect();
            tooltip.style.top = `${rect.bottom + window.scrollY}px`;
            tooltip.style.left = `${rect.left + window.scrollX}px`;
        });
        input.addEventListener('mouseleave', function() {
            document.querySelector('.tooltip')?.remove();
        });
    });
});