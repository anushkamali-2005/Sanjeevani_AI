// Agent workflow visualization
function renderWorkflow() {
    const container = document.getElementById('workflow-viz');

    const workflow = `
        <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap;">
            <div class="workflow-step">📡 Observe</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-step">📦 Aggregate</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-step">🧠 Reason</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-step">⚖️ Decide</div>
            <div class="workflow-arrow">→</div>
            <div class="workflow-step">⚡ Execute</div>
        </div>
        <div style="margin-top: 30px; text-align: center; color: #666;">
            <p><strong>Parallel Execution:</strong> Observe (all sources), Execute (multiple actions)</p>
            <p><strong>Conditional Routing:</strong> Priority-based, Confidence-based, Risk-based</p>
        </div>
    `;

    container.innerHTML = workflow;
}
