from fasthtml.common import *
from fh_matplotlib import matplotlib2fasthtml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import fh_frankenui.core as franken
from dataclasses import dataclass

@dataclass
class SimConfig:
    active_monthly_customers: int = int(5e6)
    customer_service_percentage: float = 0.02
    staged_rollout_percentage: int = 1
    turns_per_conversation: int = 5
    llm_calls_per_turn: int = 3
    input_tokens_per_message: int = 200
    output_tokens_per_message: int = 40
    input_cost_per_1k: float = 0.003
    output_cost_per_1k: float = 0.015
    tokens_per_minute_limit: int = 200000
    requests_per_minute_limit: int = 20
    distribution_variance: float = 0.15
    distribution_samples: int = 10000

class LLMTrafficSimulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self._calculate_base_metrics()
    
    def _calculate_base_metrics(self):
        weekly_cs_contacts = (self.config.active_monthly_customers * 
                            self.config.customer_service_percentage)
        self.conversations_per_week = weekly_cs_contacts * (self.config.staged_rollout_percentage / 100)
        
        total_turns = self.conversations_per_week * self.config.turns_per_conversation
        total_requests = total_turns * self.config.llm_calls_per_turn
        self.mean_requests_per_second = total_requests / (7 * 24 * 60 * 60)
        
        self.max_requests_per_second = self.config.requests_per_minute_limit / 60
        self.max_input_tokens_per_second = (self.config.tokens_per_minute_limit / 
                                          self.config.input_tokens_per_message / 60)
    
    def generate_traffic(self):
        np.random.seed(42)
        return np.random.normal(
            self.mean_requests_per_second,
            self.mean_requests_per_second * self.config.distribution_variance,
            int(self.config.distribution_samples)
        )
    
    def analyze_traffic(self, distribution):
        weekly_requests = np.mean(distribution) * (7 * 24 * 60 * 60)
        weekly_cost = self._calculate_costs(weekly_requests)
        
        return {
            'weekly_requests': weekly_requests,
            'weekly_cost': weekly_cost,
            'monthly_cost': weekly_cost * 4.33,
            'cost_per_conversation': weekly_cost / self.conversations_per_week
        }
    
    def _calculate_costs(self, requests_per_week):
        total_input_tokens = requests_per_week * self.config.input_tokens_per_message
        total_output_tokens = requests_per_week * self.config.output_tokens_per_message
        
        return ((total_input_tokens / 1000) * self.config.input_cost_per_1k +
                (total_output_tokens / 1000) * self.config.output_cost_per_1k)

# UI Component Functions
def create_control_input(name: str, min_val: float, max_val: float, 
                        default_val: float, step: float = 0.01):
    PARAM_DOCS = {
        'active_monthly_customers': 'Total number of monthly active customers',
        'customer_service_percentage': 'Percentage contacting customer service weekly',
        'staged_rollout_percentage': 'Percentage of customers rolled out to',
        'turns_per_conversation': 'Assistant answers per conversation',
        'llm_calls_per_turn': 'LLM API calls per response; 1 in simple case, more with agents',
        'input_tokens_per_message': 'Average input tokens per message',
        'output_tokens_per_message': 'Average output tokens per message',
        'input_cost_per_1k': 'Cost per 1000 input tokens',
        'output_cost_per_1k': 'Cost per 1000 output tokens',
        'tokens_per_minute_limit': 'API token rate limit per minute',
        'requests_per_minute_limit': 'API request rate limit per minute',
        'distribution_variance': 'Distribution spread control',
        'distribution_samples': 'Number of distribution samples'
    }

    return Div(
        Label(name.replace('_', ' ').title(), cls="font-bold text-sm"),
        P(PARAM_DOCS[name], cls="text-xs text-gray-500"),
        Div(
            Input(
                type="range", id=f"{name}-slider", name=name,
                min=str(min_val), max=str(max_val), value=str(default_val), step=str(step),
                style="width: 100%",
                hx_post="/update", hx_target="#results",
                hx_trigger="change, input delay:500ms",
                hx_include="#controls-form",
                oninput=f"document.getElementById('{name}-number').value = this.value"
            ),
            Input(
                type="number", id=f"{name}-number",
                min=str(min_val), max=str(max_val), value=str(default_val), step=str(step),
                style="width: 100px; margin-left: 10px",
                hx_post="/update", hx_target="#results",
                hx_trigger="change, input delay:500ms",
                hx_include="#controls-form",
                oninput=f"document.getElementById('{name}-slider').value = this.value"
            ),
            cls="flex items-center"
        ),
        style="margin-bottom: 0.75rem;"
    )

@matplotlib2fasthtml
def create_traffic_plot(simulator: LLMTrafficSimulator, traffic: np.ndarray):
    plt.figure(figsize=(12, 6))
    plt.hist(traffic, bins=50, density=True, alpha=0.7, color='skyblue')
    plt.title('LLM Traffic Distribution')
    plt.xlabel('Requests per Second')
    plt.ylabel('Density')
    
    kde = gaussian_kde(traffic)
    x_range = np.linspace(min(traffic), max(traffic), 200)
    plt.plot(x_range, kde(x_range), 'r-', lw=2, label='Traffic Distribution')
    
    mean = simulator.mean_requests_per_second
    std_dev = np.std(traffic)
    
    plt.axvline(x=mean, color='g', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.axvline(x=mean + std_dev, color='purple', linestyle=':', label=f'±1 StdDev: {std_dev:.2f}')
    plt.axvline(x=mean - std_dev, color='purple', linestyle=':')
    plt.axvline(x=simulator.max_requests_per_second, color='orange', linestyle='-', 
                label=f'Request Rate Limit: {simulator.max_requests_per_second:.2f}')
    plt.axvline(x=simulator.max_input_tokens_per_second, color='pink', linestyle='-',
                label=f'Token Rate Limit: {simulator.max_input_tokens_per_second:.2f}')
    plt.legend()
    return plt

def create_metrics_grid(analysis: dict):
    return Grid(
        Card(P(f"{analysis['weekly_requests']:,.0f}", cls="text-2xl font-bold"), 
             header=H3("Weekly Requests", cls="text-base")),
        Card(P(f"${analysis['weekly_cost']:,.2f}", cls="text-2xl font-bold"), 
             header=H3("Weekly Cost", cls="text-base")),
        Card(P(f"${analysis['monthly_cost']:,.2f}", cls="text-2xl font-bold"), 
             header=H3("Monthly Cost", cls="text-base")),
        Card(P(f"${analysis['cost_per_conversation']:.3f}", cls="text-2xl font-bold"), 
             header=H3("Cost per Conversation", cls="text-base")),
        columns="2",
        cls="gap-4 mb-4"
    )

# FastHTML App Setup
app, rt = fast_app(hdrs=(
    franken.Theme.orange.headers(),
    Script(defer=True, src="https://cdn.tailwindcss.com")
))

def run_simulation(config: SimConfig):
    simulator = LLMTrafficSimulator(config)
    traffic = simulator.generate_traffic()
    return simulator, traffic, simulator.analyze_traffic(traffic)

# Route Handlers

@rt("/")
def get():
    config = SimConfig()
    simulator, traffic, analysis = run_simulation(config)
        
    # Build content first
    content = Container(
        Card(
            P("""This simulator helps estimate LLM traffic patterns and costs based on your customer volumes. 
                Adjust the parameters below to see how different scenarios affect your usage and costs."""),
            header=H1("LLM Chatbot Traffic Simulator", cls="text-4xl text-gray-200 mt-1"),
            footer=Button("Reset to Defaults", cls="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded",
                         hx_get="/reset", hx_target="body"),
            cls="not-prose w-full mb-4"
        ),
        Div(
            Div(
                Card(
                    Form(
                        create_control_input("active_monthly_customers", 1e6, 20e6, config.active_monthly_customers, 100000),
                        create_control_input("customer_service_percentage", 0.01, 1.0, config.customer_service_percentage),
                        create_control_input("staged_rollout_percentage", 1, 100, config.staged_rollout_percentage, 1),
                        create_control_input("turns_per_conversation", 1, 20, config.turns_per_conversation, 1),
                        create_control_input("llm_calls_per_turn", 1, 10, config.llm_calls_per_turn, 1),
                        create_control_input("input_tokens_per_message", 50, 1000, config.input_tokens_per_message, 10),
                        create_control_input("output_tokens_per_message", 10, 200, config.output_tokens_per_message, 5),
                        create_control_input("input_cost_per_1k", 0.001, 0.01, config.input_cost_per_1k, 0.001),
                        create_control_input("output_cost_per_1k", 0.001, 0.05, config.output_cost_per_1k, 0.001),
                        create_control_input("tokens_per_minute_limit", 50000, 1000000, config.tokens_per_minute_limit, 10000),
                        create_control_input("requests_per_minute_limit", 1, 300, config.requests_per_minute_limit, 20),
                        create_control_input("distribution_variance", 0.01, 0.5, config.distribution_variance, 0.01),
                        create_control_input("distribution_samples", 1000, 50000, config.distribution_samples, 1000),
                        id="controls-form"
                    ),
                    header=H2("Controls", cls="card-title text-base"),
                    cls="not-prose w-full p-4"
                ),
                cls="col-span-4"
            ),
            Div(
                Card(
                    Div(
                        create_metrics_grid(analysis),
                        Div(create_traffic_plot(simulator, traffic)),
                        id="results"
                    ),
                    header=H2("Results", cls="card-title"),
                    cls="not-prose w-full"
                ),
                cls="col-span-8"
            ),
            cls="grid grid-cols-12 gap-8",
        )
    )

    # Let FastHTML handle the response with headers
    return Title("LLM Chatbot Traffic Simulator"), content

@rt("/reset")
def reset():
    # Use 303 to force a GET request with fresh defaults
    return RedirectResponse("/", status_code=303)

@rt("/update")
async def post(request):
    form = await request.form()
    config = SimConfig(**{k: float(form.get(k, getattr(SimConfig(), k))) for k in SimConfig.__annotations__})
    simulator, traffic, analysis = run_simulation(config)
    
    return Div(
        create_metrics_grid(analysis),
        Div(create_traffic_plot(simulator, traffic)),
        id="results"
    )

serve()