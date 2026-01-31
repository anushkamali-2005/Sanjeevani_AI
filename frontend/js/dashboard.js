// Main dashboard logic
document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-agent-btn');
    const refreshBtn = document.getElementById('refresh-btn');

    runBtn.addEventListener('click', async () => {
        runBtn.disabled = true;
        runBtn.textContent = '⏳ Running...';

        try {
            const result = await api.runAgent();
            displayResults(result);
        } catch (error) {
            console.error('Error running agent:', error);
            alert('Failed to run agent. Check console for details.');
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = '▶ Run Agent Workflow';
        }
    });

    refreshBtn.addEventListener('click', async () => {
        await updateStatus();
    });

    // Initial status update
    updateStatus();
});

async function updateStatus() {
    try {
        const status = await api.getAgentStatus();
        document.getElementById('observer-status').style.color = '#4CAF50';
        document.getElementById('reasoning-status').style.color = '#4CAF50';
        document.getElementById('decision-status').style.color = '#4CAF50';
        document.getElementById('executor-status').style.color = '#4CAF50';
    } catch (error) {
        console.error('Status update failed:', error);
    }
}

function displayResults(result) {
    // Display decision
    const decisionContainer = document.getElementById('decision-container');
    if (result.decision) {
        const confidence = (result.confidence * 100).toFixed(1);
        const confidenceClass = result.confidence >= 0.9 ? 'high' : result.confidence >= 0.7 ? 'medium' : 'low';

        decisionContainer.innerHTML = `
            <div class="decision-summary">
                <p><strong>Root Cause:</strong> ${result.decision.root_cause || 'Analyzing...'}</p>
                <p><strong>Confidence:</strong> <span class="confidence-badge confidence-${confidenceClass}">${confidence}%</span></p>
                <p><strong>Affected Tickets:</strong> ${result.decision.ticket_ids?.length || 0}</p>
            </div>
        `;
    }

    // Display explanation
    const explanationContainer = document.getElementById('explanation-container');
    explanationContainer.innerHTML = `<p>${result.explanation || 'No explanation available.'}</p>`;

    // Display actions
    const actionsContainer = document.getElementById('actions-container');
    if (result.decision?.proposed_actions) {
        actionsContainer.innerHTML = result.decision.proposed_actions.map(action => `
            <div class="action-item">
                <strong>${action.action_type}</strong>
                <p>Risk: ${action.risk_level} | Approval: ${action.requires_approval ? 'Required' : 'Not Required'}</p>
                <p>${action.reasoning}</p>
            </div>
        `).join('');
    }

    // Display uncertainties
    const uncertaintiesContainer = document.getElementById('uncertainties-container');
    if (result.decision) {
        uncertaintiesContainer.innerHTML = `
            <h3>Assumptions:</h3>
            <ul>${result.decision.assumptions?.map(a => `<li>${a}</li>`).join('') || '<li>None</li>'}</ul>
            <h3>Uncertainties:</h3>
            <ul>${result.decision.uncertainty_factors?.map(u => `<li>${u}</li>`).join('') || '<li>None</li>'}</ul>
        `;
    }

    // Render workflow
    renderWorkflow();
}
