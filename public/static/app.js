/**
 * Darukaa.Earth — Frontend Application Logic
 * Handles chat interaction, structured form submission, slot synchronization,
 * transparent multi-metric recommendation cards, alternative recommendation cards,
 * and sources trace display.
 */

// --- State ---
let sessionId = localStorage.getItem('darukaa_session_id') || null;
let currentSlots = {};

// --- DOM References ---
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');
const slotCount = document.getElementById('slot-count');
const activeSlotsListEl = document.getElementById('active-slots-list');
const structuredForm = document.getElementById('structured-form');
const formToggle = document.getElementById('form-toggle');
const formToggleIcon = document.getElementById('form-toggle-icon');
const clearSlotsBtn = document.getElementById('clear-slots-btn');
const formSubmitBtn = document.getElementById('form-submit-btn');

// Stats
const statChunks = document.getElementById('stat-chunks');
const statInterventions = document.getElementById('stat-interventions');
const statVariables = document.getElementById('stat-variables');

// API base URL
const API_BASE = '';  // same origin


// --- Initialize ---
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    setupEventListeners();
});


function setupEventListeners() {
    // Chat form submission
    chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        sendChatMessage();
    });

    // Enter to send (Shift+Enter for newline)
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });

    // Form toggle
    formToggle.addEventListener('click', toggleStructuredForm);
    formToggle.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleStructuredForm();
        }
    });

    // Structured form submission
    structuredForm.addEventListener('submit', (e) => {
        e.preventDefault();
        submitStructuredForm();
    });

    // Clear/reset session
    clearSlotsBtn.addEventListener('click', resetSession);
}


function toggleStructuredForm() {
    const isCollapsed = structuredForm.classList.toggle('collapsed');
    formToggleIcon.classList.toggle('rotated', !isCollapsed);
    formToggle.setAttribute('aria-expanded', String(!isCollapsed));
}


// --- Stats ---
async function loadStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        if (res.ok) {
            const data = await res.json();
            if (statChunks) statChunks.textContent = data.knowledge_base?.total_chunks || '23';
            if (statInterventions) statInterventions.textContent = data.relationship_graph?.total_interventions || '12';
            if (statVariables) statVariables.textContent = data.tracked_variables || 9;
        }
    } catch (err) {
        console.error('Failed to load stats:', err);
    }
}


// --- Chat ---
async function sendChatMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message to UI
    addMessage(message, 'user');
    chatInput.value = '';
    chatSendBtn.disabled = true;

    const loadingEl = addLoading();

    const payload = {
        session_id: sessionId,
        message: message
    };
    console.log('[Darukaa Chat] Sending payload:', payload);

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        loadingEl.remove();

        if (res.ok) {
            const data = await res.json();
            sessionId = data.session_id;
            if (sessionId) {
                localStorage.setItem('darukaa_session_id', sessionId);
            }
            handleChatResponse(data);
        } else {
            const errData = await res.json().catch(() => ({ detail: 'Unknown error' }));
            addMessage(`Error: ${errData.detail || 'Something went wrong.'}`, 'system');
        }
    } catch (err) {
        loadingEl.remove();
        addMessage(`Connection error: ${err.message}. Is the server running?`, 'system');
    }

    chatSendBtn.disabled = false;
    chatInput.focus();
}


function handleChatResponse(data) {
    // Update active slots display
    if (data.slots) {
        updateSlotDisplay(data.slots);
    }

    // Show slot update notice
    if (data.slot_updated) {
        addSlotUpdateNotice(data.slot_updated);
    }

    const allSources = (data.sources && data.sources.length > 0) ? data.sources : (data.sources_used || []);

    const finishDisplay = () => {
        // Render primary recommendation card if present
        if (data.recommendation) {
            renderRecommendationCard(data.recommendation, allSources);

            // Render alternative recommendation if present
            if (data.alternative_recommendation) {
                renderAlternativeCard(data.alternative_recommendation, data.why_alternative);
            }
        } else if (allSources.length > 0 || data.source_status === 'unverified_fallback' || data.verified === false) {
            // Render sources panel for live web results, internal docs, or unverified AI fallback notice
            const standaloneSources = document.createElement('div');
            standaloneSources.className = 'standalone-sources-wrapper';
            standaloneSources.style.marginTop = '8px';
            standaloneSources.innerHTML = renderSourcesPanel(allSources, data.source_status, data.verified);
            chatMessages.appendChild(standaloneSources);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    };

    // Autoregressive generation of response text
    if (data.response_text) {
        addMessageAutoregressive(data.response_text, finishDisplay);
    } else {
        finishDisplay();
    }
}


// --- Structured Form ---
async function submitStructuredForm() {
    const payload = {};

    const soc = document.getElementById('form-soc').value;
    if (soc) payload.soil_organic_carbon_pct = parseFloat(soc);

    const ph = document.getElementById('form-ph').value;
    if (ph) payload.soil_ph = parseFloat(ph);

    const moisture = document.getElementById('form-moisture').value;
    if (moisture) payload.soil_moisture = moisture;

    const landuse = document.getElementById('form-landuse').value;
    if (landuse) payload.land_use_type = landuse;

    const rainfall = document.getElementById('form-rainfall').value;
    if (rainfall) payload.rainfall_pattern = rainfall;

    const climate = document.getElementById('form-climate').value;
    if (climate) payload.region_climate_zone = climate;

    const species = document.getElementById('form-species').value;
    if (species) payload.species_richness_observation = species;

    const pollution = document.getElementById('form-pollution').value;
    if (pollution) payload.pollution_or_deforestation_pressure = pollution;

    const filledCount = Object.keys(payload).length;
    if (filledCount < 2) {
        addMessage('Please fill in at least 2 environmental variables for structured analysis.', 'system');
        return;
    }

    const summaryParts = Object.entries(payload).map(([k, v]) =>
        `• ${formatSlotName(k)}: ${v}`
    );
    addMessage(`Analysing structured variables:\n${summaryParts.join('\n')}`, 'user');

    formSubmitBtn.disabled = true;
    const loadingEl = addLoading();

    try {
        const res = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        loadingEl.remove();

        if (res.ok) {
            const data = await res.json();

            if (data.slots_used) {
                updateSlotDisplay(data.slots_used);
            }

            if (data.reasoning_trace) {
                addMessage(data.reasoning_trace, 'system');
            }

            if (data.recommendation) {
                const allSources = (data.sources && data.sources.length > 0) ? data.sources : (data.sources_used || []);
                renderRecommendationCard(data.recommendation, allSources);
                if (data.alternative_recommendation) {
                    renderAlternativeCard(data.alternative_recommendation, data.why_alternative);
                }
            }
        } else {
            const errData = await res.json().catch(() => ({}));
            const errorMsg = errData.error || errData.detail || 'Additional analysis is temporarily unavailable.';
            addMessage(`Analysis notice: ${errorMsg}`, 'system');
        }
    } catch (err) {
        loadingEl.remove();
        addMessage('Analysis notice: Additional analysis is temporarily unavailable.', 'system');
    }

    formSubmitBtn.disabled = false;
}


// --- UI Helpers ---

function formatMessageHtml(rawText) {
    const paragraphs = rawText.split('\n');
    return paragraphs.map(p => {
        if (!p.trim()) return '';
        const formatted = p
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/•/g, '&bull;')
            .replace(/\[(DOC_\d+|WEB_\d+|DATA_\d+)\]/g, '<span class="citation-badge">[$1]</span>');
        return `<p>${formatted}</p>`;
    }).join('');
}


function addMessage(text, type) {
    const div = document.createElement('div');
    div.className = `message ${type}-message`;

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = formatMessageHtml(text);

    div.appendChild(content);
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}


function addMessageAutoregressive(text, onComplete) {
    const div = document.createElement('div');
    div.className = 'message system-message';

    const content = document.createElement('div');
    content.className = 'message-content';
    div.appendChild(content);

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Tokenize text into chunks (words and whitespaces)
    const tokens = text.match(/([^\s]+|\s+)/g) || [text];
    let currentIdx = 0;
    let accumulatedText = '';

    const cursor = document.createElement('span');
    cursor.className = 'typing-cursor';
    cursor.textContent = '▌';

    function step() {
        if (currentIdx < tokens.length) {
            accumulatedText += tokens[currentIdx];
            currentIdx++;

            content.innerHTML = formatMessageHtml(accumulatedText);
            const lastP = content.querySelector('p:last-of-type') || content;
            lastP.appendChild(cursor);

            chatMessages.scrollTop = chatMessages.scrollHeight;
            setTimeout(step, 14); // ~14ms per word for natural, fluid generation
        } else {
            content.innerHTML = formatMessageHtml(accumulatedText);
            cursor.remove();
            chatMessages.scrollTop = chatMessages.scrollHeight;
            if (onComplete) onComplete();
        }
    }

    step();
    return div;
}


function addLoading() {
    const div = document.createElement('div');
    div.className = 'message system-message loading-message';
    div.innerHTML = `<div class="loading-dots"><span></span><span></span><span></span></div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}


function addSlotUpdateNotice(noticeText) {
    const div = document.createElement('div');
    div.className = 'slot-update-notice';
    div.textContent = `📝 Updated: ${noticeText} — recommendation revised.`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


function renderRecommendationCard(rec, sources) {
    const card = document.createElement('div');
    card.className = 'recommendation-card';

    const confidence = rec.confidence || 'medium';
    const confidenceClass = `confidence-${confidence}`;

    // Format matched variables summary
    const matchedVarsText = (rec.matched_variables && rec.matched_variables.length > 0)
        ? rec.matched_variables.map(v => formatSlotName(v)).join(' + ')
        : (rec.connected_variables || []).map(v => formatSlotName(v)).join(' + ');

    card.innerHTML = `
        <div class="rec-card-header">
            <div>
                <h3>${escapeHtml(rec.recommendation)}</h3>
                <div class="rec-matched-pill">
                    Based on: ${escapeHtml(matchedVarsText || 'Environmental parameters')} (${rec.overlap_count || 2} graph overlaps)
                </div>
            </div>
            <span class="rec-badge ${confidenceClass}">${confidence.toUpperCase()} CONFIDENCE</span>
        </div>
        <div class="rec-card-body">
            <!-- Mechanism (collapsible) -->
            <div class="rec-mechanism">
                <button class="rec-mechanism-toggle" onclick="toggleMechanism(this)" aria-expanded="false">
                    <span>Why this works (Scientific Mechanism)</span>
                    <span class="toggle-icon">▼</span>
                </button>
                <div class="rec-mechanism-content">
                    ${escapeHtml(rec.mechanism)}
                </div>
            </div>

            <!-- Co-benefits and Trade-offs (Two-column layout) -->
            ${renderCoBenefitsAndTradeoffs(rec.co_benefits, rec.trade_offs)}

            <!-- Metrics table -->
            ${renderMetricsTable(rec.metrics_impacted || [])}

            <!-- Meta row -->
            <div class="rec-meta-row">
                <div class="rec-meta-item">
                    <span class="rec-meta-label">Time Horizon:</span>
                    <span class="rec-meta-value">${formatTimeHorizon(rec.time_horizon)}</span>
                </div>
            </div>

            <!-- Connected variables -->
            ${renderConnectedVars(rec.connected_variables || rec.matched_variables || [])}

            <!-- Source citation -->
            <div class="rec-source">
                <span class="rec-source-label">Primary Sources: </span>
                ${escapeHtml(rec.source)}
            </div>

            <!-- Sources panel (collapsible, integrated inside card) -->
            ${renderSourcesPanel(sources)}
        </div>
    `;

    chatMessages.appendChild(card);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


function renderAlternativeCard(altRec, whyAlt) {
    const card = document.createElement('div');
    card.className = 'recommendation-card alternative-card';

    const confidence = altRec.confidence || 'low';
    const confidenceClass = `confidence-${confidence}`;

    card.innerHTML = `
        <div class="rec-card-header">
            <div>
                <span class="rec-alt-label">ALTERNATIVE INTERVENTION</span>
                <h3>${escapeHtml(altRec.recommendation)}</h3>
            </div>
            <span class="rec-badge ${confidenceClass}">${confidence.toUpperCase()} CONFIDENCE</span>
        </div>
        <div class="rec-card-body">
            ${whyAlt ? `<p class="why-alt-text"><strong>Why consider this alternative:</strong> ${escapeHtml(whyAlt)}</p>` : ''}
            <div class="rec-mechanism-content" style="display:block; padding: 8px 0;">
                ${escapeHtml(altRec.mechanism)}
            </div>
            ${renderMetricsTable(altRec.metrics_impacted || [])}
        </div>
    `;

    chatMessages.appendChild(card);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}


function renderCoBenefitsAndTradeoffs(coBenefits, tradeOffs) {
    if ((!coBenefits || coBenefits.length === 0) && (!tradeOffs || tradeOffs.length === 0)) return '';

    const benefitsList = (coBenefits || []).map(b => `<li class="benefit">${escapeHtml(b)}</li>`).join('');
    const tradeoffsList = (tradeOffs || []).map(t => `<li class="tradeoff">${escapeHtml(t)}</li>`).join('');

    return `
        <div class="rec-cobenefits-grid">
            <div>
                <div class="rec-col-title positive">✓ Co-Benefits</div>
                <ul class="rec-list">${benefitsList || '<li>Ecosystem resilience enhancements</li>'}</ul>
            </div>
            <div>
                <div class="rec-col-title negative">⚠ Trade-Offs / Caveats</div>
                <ul class="rec-list">${tradeoffsList || '<li>Requires seasonal monitoring and establishment care</li>'}</ul>
            </div>
        </div>
    `;
}


function renderMetricsTable(metrics) {
    if (!metrics || metrics.length === 0) return '';

    let rows = metrics.map(m => {
        const arrow = m.direction === 'increase' ? '↑' : '↓';
        const dirClass = `direction-${m.direction}`;
        return `
            <tr>
                <td>${formatMetricName(m.metric)}</td>
                <td class="${dirClass}">${arrow} ${m.direction}</td>
                <td>${escapeHtml(m.magnitude)}</td>
            </tr>
        `;
    }).join('');

    return `
        <table class="rec-metrics-table">
            <thead>
                <tr><th>Metric</th><th>Direction</th><th>Magnitude</th></tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}


function renderConnectedVars(vars) {
    if (!vars || vars.length === 0) return '';

    const tags = vars.map(v =>
        `<span class="var-tag">${formatSlotName(v)}</span>`
    ).join('');

    return `
        <div>
            <span class="rec-meta-label">Connected Variables: </span>
            <div class="rec-connected-vars" style="margin-top: 4px;">${tags}</div>
        </div>
    `;
}


function renderSourcesPanel(sources, sourceStatus, isVerified) {
    const isUnverified = sourceStatus === 'unverified_fallback' || isVerified === false || (sources && sources.some(s => s.source_type === 'llm_knowledge'));

    if (isUnverified && (!sources || sources.length === 0 || sources.every(s => s.source_type === 'llm_knowledge'))) {
        return `
            <div class="unverified-source-banner">
                <div class="unverified-source-header">
                    <span class="unverified-badge">⚠️ AI-generated</span>
                    <span class="unverified-title">No verified source found</span>
                </div>
                <div class="unverified-desc">This answer was generated from general model knowledge — no verified internal documents or live web sources were found for this response.</div>
            </div>
        `;
    }

    if (!sources || sources.length === 0) return '';

    // STRICT PROVENANCE ORDERING: 1. Documents first, 2. Web sources second, 3. Empirical datasets third, 4. AI fallback last
    const typeOrder = { 'document': 0, 'web': 1, 'web_search': 1, 'structured_data': 2, 'llm_knowledge': 3 };
    const sortedSources = [...sources].sort((a, b) => {
        const aType = a.source_type || a.type || 'document';
        const bType = b.source_type || b.type || 'document';
        return (typeOrder[aType] ?? 99) - (typeOrder[bType] ?? 99);
    });

    const sourceItems = sortedSources.map((s) => {
        const type = s.source_type || s.type || 'document';
        const tagBadge = s.id ? `<span class="source-tag-id">[${escapeHtml(s.id)}]</span>` : '';

        if (type === 'llm_knowledge') {
            return `
                <div class="source-card source-card-unverified">
                    <div class="source-card-top">
                        <span class="source-badge badge-unverified">⚠️ AI Knowledge</span>
                        ${tagBadge}
                    </div>
                    <div class="source-card-title">General Model Knowledge</div>
                    <div class="source-card-sub">Answered from pre-trained model weights without verified external grounding.</div>
                </div>
            `;
        } else if (type === 'web' || type === 'web_search') {
            // Live Web Source (ChatGPT / Claude style clean tile — no wall of text)
            const title = s.title || 'Live Web Source';
            const url = s.url || s.source_url || '#';
            let domain = '';
            try {
                if (url && url.startsWith('http')) {
                    domain = new URL(url).hostname.replace('www.', '');
                }
            } catch (e) {
                domain = '';
            }
            const snippet = s.detail || s.excerpt || s.snippet || '';

            return `
                <div class="source-card source-card-web" onclick="if('${escapeHtml(url)}' !== '#') window.open('${escapeHtml(url)}', '_blank')">
                    <div class="source-card-top">
                        <span class="source-domain-chip">
                            <span class="source-icon">🌐</span>
                            <span class="source-domain-name">${escapeHtml(domain || 'Web Source')}</span>
                        </span>
                        ${tagBadge}
                        ${url && url !== '#' ? `
                            <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="source-external-link" onclick="event.stopPropagation();" title="Open source in new tab">↗</a>
                        ` : ''}
                    </div>
                    <div class="source-card-title" title="${escapeHtml(title)}">${escapeHtml(title)}</div>
                    ${snippet ? `
                        <details class="source-inspect-drawer" onclick="event.stopPropagation();">
                            <summary class="source-inspect-summary">Inspect Evidence</summary>
                            <div class="source-inspect-content">${escapeHtml(snippet)}</div>
                        </details>
                    ` : ''}
                </div>
            `;
        } else if (type === 'structured_data') {
            // Structured CSV dataset
            const fileName = s.file_name || s.file || s.title || 'Structured Dataset';
            const detailText = s.detail || s.text || '';
            return `
                <div class="source-card source-card-dataset">
                    <div class="source-card-top">
                        <span class="source-domain-chip data-chip">
                            <span class="source-icon">📊</span>
                            <span class="source-domain-name">Empirical Dataset</span>
                        </span>
                        ${tagBadge}
                    </div>
                    <div class="source-card-title">${escapeHtml(fileName)}</div>
                    ${detailText ? `
                        <details class="source-inspect-drawer">
                            <summary class="source-inspect-summary">Inspect Evidence</summary>
                            <div class="source-inspect-content">${escapeHtml(detailText)}</div>
                        </details>
                    ` : ''}
                </div>
            `;
        } else {
            // Internal Document (IPCC AR6 WGII Report / Curated Knowledge Base) — ALWAYS LISTED FIRST
            const title = s.title || s.document_title || 'Internal Knowledge Document';
            const fileName = s.file_name || s.file || (s.source ? s.source : 'document.pdf');
            const pageNum = s.page_number || s.page || (s.id && s.id.includes('_p') ? s.id.split('_p')[1].split('_')[0] : '');
            const pageDisplay = pageNum ? (String(pageNum).startsWith('Page') ? pageNum : `Page ${pageNum}`) : '';
            const excerpt = s.excerpt || s.text || '';

            return `
                <div class="source-card source-card-doc">
                    <div class="source-card-top">
                        <span class="source-domain-chip doc-chip">
                            <span class="source-icon">📄</span>
                            <span class="source-domain-name">${escapeHtml(fileName)}</span>
                        </span>
                        ${tagBadge}
                        ${pageDisplay ? `<span class="source-page-pill">${escapeHtml(pageDisplay)}</span>` : ''}
                    </div>
                    <div class="source-card-title" title="${escapeHtml(title)}">${escapeHtml(title)}</div>
                    ${excerpt ? `
                        <details class="source-inspect-drawer">
                            <summary class="source-inspect-summary">Inspect Evidence</summary>
                            <div class="source-inspect-content">${escapeHtml(excerpt)}</div>
                        </details>
                    ` : ''}
                </div>
            `;
        }
    }).join('');

    return `
        <div class="sources-panel">
            <button class="sources-toggle" onclick="toggleSources(this)" aria-expanded="true">
                <span>Sources & Citations (${sortedSources.length} sources)</span>
                <span class="toggle-icon">▲</span>
            </button>
            <div class="sources-content" style="display: block;">
                <div class="sources-cards-grid">${sourceItems}</div>
            </div>
        </div>
    `;
}


// --- Slot Display ---

function updateSlotDisplay(slots) {
    currentSlots = slots;

    const known = Object.entries(slots).filter(([k, v]) => v !== null && v !== undefined && k !== 'geo_lat' && k !== 'geo_lon');
    const total = 9;
    slotCount.textContent = `${known.length} / ${total}`;

    if (known.length === 0) {
        activeSlotsListEl.innerHTML = '<p class="empty-state">No variables identified yet. Start chatting or use the form below.</p>';
        return;
    }

    activeSlotsListEl.innerHTML = known.map(([k, v]) => `
        <div class="slot-item">
            <span class="slot-name">${formatSlotName(k)}</span>
            <span class="slot-value">${escapeHtml(String(v))}</span>
        </div>
    `).join('');

    syncFormWithSlots(slots);
}


function syncFormWithSlots(slots) {
    if (slots.soil_organic_carbon_pct != null)
        document.getElementById('form-soc').value = slots.soil_organic_carbon_pct;
    if (slots.soil_ph != null)
        document.getElementById('form-ph').value = slots.soil_ph;
    if (slots.soil_moisture)
        document.getElementById('form-moisture').value = slots.soil_moisture;
    if (slots.land_use_type)
        document.getElementById('form-landuse').value = slots.land_use_type;
    if (slots.rainfall_pattern)
        document.getElementById('form-rainfall').value = slots.rainfall_pattern;
    if (slots.region_climate_zone)
        document.getElementById('form-climate').value = slots.region_climate_zone;
    if (slots.species_richness_observation)
        document.getElementById('form-species').value = slots.species_richness_observation;
    if (slots.pollution_or_deforestation_pressure)
        document.getElementById('form-pollution').value = slots.pollution_or_deforestation_pressure;
}


function resetSession() {
    sessionId = null;
    localStorage.removeItem('darukaa_session_id');
    currentSlots = {};
    slotCount.textContent = '0 / 9';
    activeSlotsListEl.innerHTML = '<p class="empty-state">No variables identified yet. Start chatting or use the form below.</p>';

    structuredForm.reset();

    chatMessages.innerHTML = `
        <div class="message system-message">
            <div class="message-content">
                <p>Welcome to <strong>Darukaa.Earth</strong>. I'm your evidence-based biodiversity intelligence advisor.</p>
                <p>Tell me about your land — its location, soil conditions, land use type, and rainfall pattern — and I'll generate auditable, scientifically grounded recommendations.</p>
                <p class="hint">Try: <em>"I have a wheat farm in Rajasthan with low rainfall and soil organic carbon around 0.3%"</em></p>
            </div>
        </div>
    `;
}


// --- Toggle Functions (global for onclick) ---

window.toggleMechanism = function(btn) {
    const content = btn.nextElementSibling;
    const icon = btn.querySelector('.toggle-icon');
    const isExpanded = content.classList.toggle('expanded');
    icon.classList.toggle('rotated', isExpanded);
    btn.setAttribute('aria-expanded', String(isExpanded));
};

window.toggleSources = function(btn) {
    const content = btn.nextElementSibling;
    const icon = btn.querySelector('.toggle-icon');
    const isExpanded = content.classList.toggle('expanded');
    icon.classList.toggle('rotated', isExpanded);
    btn.setAttribute('aria-expanded', String(isExpanded));
};

window.toggleAlternative = function(btn) {
    const content = btn.nextElementSibling;
    const icon = btn.querySelector('.toggle-icon');
    const isExpanded = content.classList.toggle('expanded');
    icon.classList.toggle('rotated', isExpanded);
    btn.setAttribute('aria-expanded', String(isExpanded));
};


// --- Formatting Helpers ---

function formatSlotName(name) {
    const labels = {
        'soil_organic_carbon_pct': 'Soil Organic Carbon',
        'soil_ph': 'Soil pH',
        'soil_moisture': 'Soil Moisture',
        'land_use_type': 'Land Use Type',
        'rainfall_pattern': 'Rainfall Pattern',
        'region_climate_zone': 'Climate Zone',
        'species_richness_observation': 'Species Richness',
        'pollution_or_deforestation_pressure': 'Pollution/Deforestation',
        'geo_lat': 'Latitude',
        'geo_lon': 'Longitude',
    };
    return labels[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}


function formatMetricName(name) {
    return name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}


function formatTimeHorizon(th) {
    const labels = {
        'short_term': 'Short-term (< 1 year)',
        'short_to_medium_term': 'Short to Medium-term (1–3 years)',
        'medium_term': 'Medium-term (2–5 years)',
        'medium_to_long_term': 'Medium to Long-term (3–10 years)',
        'long_term': 'Long-term (5+ years)',
    };
    return labels[th] || th.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}


function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
