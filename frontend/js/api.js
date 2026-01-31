// API client for backend communication
class API {
    constructor() {
        this.baseURL = window.location.origin;
    }

    async runAgent() {
        const response = await fetch(`${this.baseURL}/api/agent/run`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        return await response.json();
    }

    async getTickets() {
        const response = await fetch(`${this.baseURL}/api/tickets`);
        return await response.json();
    }

    async getMerchants() {
        const response = await fetch(`${this.baseURL}/api/merchants`);
        return await response.json();
    }

    async getAgentStatus() {
        const response = await fetch(`${this.baseURL}/api/agent/status`);
        return await response.json();
    }
}

const api = new API();
