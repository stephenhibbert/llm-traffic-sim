from math import ceil
from fasthtml.common import *
from fh_matplotlib import matplotlib2fasthtml
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import fh_frankenui.core as franken
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimConfig:
    # Traffic Parameters
    active_weekly_customers: int = 10000000
    customer_service_percentage: int = 2
    staged_rollout_percentage: int = 10
    distribution_variance: float = 0.15
    distribution_samples: int = 10000
    
    # LLM Parameters
    turns_per_conversation: int = 5
    llm_calls_per_turn: int = 3
    input_tokens_per_message: int = 100
    output_tokens_per_message: int = 20
    input_cost_per_1k: float = 0.003
    output_cost_per_1k: float = 0.015
    tokens_per_minute_limit: int = 200000
    requests_per_minute_limit: int = 20
    
    # Guardrail Parameters
    input_text_units: int = 25  # Text units per input message (1 unit = 1000 chars)
    output_text_units: int = 5  # Text units per output message

    # Guardrails quotas (per second)
    content_filter_quota: float = 25
    denied_topics_quota: float = 25
    pii_filter_quota: float = 25
    contextual_grounding_quota: float = 53
    guardrails_request_quota: float = 25

    # Guardrail enables
    content_filter_enabled: bool = True
    denied_topics_enabled: bool = False
    pii_filter_enabled: bool = True
    contextual_grounding_enabled: bool = True


class SimulationState:
    def __init__(self):
        self.config = SimConfig()
        self.traffic: Optional[np.ndarray] = None
        self.simulator: Optional[LLMTrafficSimulator] = None
        self.analysis: Optional[dict] = None

state = SimulationState()

class LLMTrafficSimulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self._calculate_base_metrics()
    
    def _calculate_base_metrics(self):
        # Existing metrics calculation...
        weekly_cs_contacts = (self.config.active_weekly_customers * 
                            (self.config.customer_service_percentage / 100))
        self.conversations_per_week = weekly_cs_contacts * (self.config.staged_rollout_percentage / 100)
        
        total_turns = self.conversations_per_week * self.config.turns_per_conversation
        total_requests = total_turns * self.config.llm_calls_per_turn
        self.mean_requests_per_second = total_requests / (7 * 24 * 60 * 60)
        self.mean_turns_per_second = total_turns / (7 * 24 * 60 * 60)
        
        # LLM limits
        self.max_requests_per_second = self.config.requests_per_minute_limit / 60
        self.max_input_tokens_per_second = (self.config.tokens_per_minute_limit / 
                                          self.config.input_tokens_per_message / 60)
        
        
        self.text_units_per_request = {
            'content_filter': self.config.input_text_units,
            'denied_topics': self.config.input_text_units,
            'pii_filter': self.config.input_text_units,
            'contextual_grounding': self.config.output_text_units
        }
            
        # Store quotas for plotting
        self.guardrails_quotas = {
            'content_filter': self.config.content_filter_quota,
            'denied_topics': self.config.denied_topics_quota,
            'pii_filter': self.config.pii_filter_quota,
            'contextual_grounding': self.config.contextual_grounding_quota,
            'requests': self.config.guardrails_request_quota
        }

    def generate_traffic(self):
        np.random.seed(42)
        return np.random.normal(
            self.mean_turns_per_second,
            self.mean_turns_per_second * self.config.distribution_variance,
            int(self.config.distribution_samples)
        )
    
    def _calculate_costs(self, requests_per_week):
        total_input_tokens = requests_per_week * self.config.input_tokens_per_message
        total_output_tokens = requests_per_week * self.config.output_tokens_per_message
        
        return ((total_input_tokens / 1000) * self.config.input_cost_per_1k +
                (total_output_tokens / 1000) * self.config.output_cost_per_1k)
    
    def _calculate_text_units(self, requests_per_week):
        text_units = {}
        for guardrail, text_units_per_request in self.text_units_per_request.items():
            if getattr(self.config, f"{guardrail}_enabled", True):
                text_units[guardrail] = requests_per_week * max(text_units_per_request, 1)
        return text_units

    def _calculate_guardrails_costs(self, requests_per_week):
        # Pricing per 1000 text units
        prices = {
            'content_filter': 0.75,
            'denied_topics': 1.0,
            'contextual_grounding': 0.1,
            'pii_filter': 0.1
        }
        
        total_cost = 0
        costs_breakdown = {}
        text_units = self._calculate_text_units(requests_per_week)
        
        for guardrail, price in prices.items():
            if guardrail in text_units:
                cost = (text_units[guardrail] / 1000) * price
                total_cost += cost
                costs_breakdown[guardrail] = cost
            else:
                costs_breakdown[guardrail] = 0
        
        return total_cost, costs_breakdown

    def analyze_traffic(self, distribution):
        weekly_requests = np.mean(distribution) * (7 * 24 * 60 * 60)
        weekly_llm_requests = weekly_requests * self.config.llm_calls_per_turn
        llm_cost = self._calculate_costs(weekly_llm_requests)
        text_units = self._calculate_text_units(weekly_requests)
        guardrails_cost, guardrails_breakdown = self._calculate_guardrails_costs(weekly_requests)
        
        return {
            'weekly_requests': weekly_requests,
            'weekly_llm_requests': weekly_llm_requests,
            'weekly_llm_cost': llm_cost,
            'weekly_guardrails_cost': guardrails_cost,
            'guardrails_breakdown': guardrails_breakdown,
            'text_units': text_units,
            'monthly_llm_cost': llm_cost * 4.33,
            'monthly_guardrails_cost': guardrails_cost * 4.33,
            'llm_cost_per_conversation': llm_cost / self.conversations_per_week,
            'guardrails_cost_per_conversation': guardrails_cost / self.conversations_per_week,
        }

def create_control_input(name: str, min_val: float, max_val: float, 
                        default_val: float, step: float = 0.01):
    PARAM_DOCS = {
        # Traffic Parameters
        'active_weekly_customers': 'Total number of unique customers using your service each week',
        'customer_service_percentage': 'Percentage of customers that contact customer service each week',
        'staged_rollout_percentage': 'Percentage of eligible conversations using the AI assistant',
        'distribution_variance': 'Controls how much traffic varies from the mean (higher = more variance)',
        'distribution_samples': 'Number of data points to simulate (higher = smoother distribution)',
        
        # LLM Parameters
        'turns_per_conversation': 'Average number of back-and-forth messages in each conversation',
        'llm_calls_per_turn': 'LLM API calls needed per assistant response (1 for simple completion, more for agents)',
        'input_tokens_per_message': 'Average number of tokens in each user message plus system prompts',
        'output_tokens_per_message': 'Average number of tokens in each assistant response',
        'input_cost_per_1k': 'Cost in USD per 1000 input tokens',
        'output_cost_per_1k': 'Cost in USD per 1000 output tokens',
        'tokens_per_minute_limit': 'Maximum tokens allowed per minute by the API',
        'requests_per_minute_limit': 'Maximum requests allowed per minute by the API',
        
        # Guardrail Parameters
        'content_filter_quota': 'Maximum content filter text units processed per second (1000 chars = 1 text unit)',
        'denied_topics_quota': 'Maximum denied topics text units processed per second (1000 chars = 1 text unit)',
        'pii_filter_quota': 'Maximum PII filter text units processed per second (1000 chars = 1 text unit)',
        'contextual_grounding_quota': 'Maximum contextual grounding text units processed per second (1000 chars = 1 text unit)',
        'guardrails_request_quota': 'Maximum guardrails API requests allowed per second',
        'input_text_units': 'Number of text units per input message (1 text unit = 1000 characters)',
        'output_text_units': 'Number of text units per output message (1 text unit = 1000 characters)',
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
                hx_include="form",
                oninput=f"document.getElementById('{name}-number').value = this.value"
            ),
            Input(
                type="number", id=f"{name}-number",
                min=str(min_val), max=str(max_val), value=str(default_val), step=str(step),
                style="width: 100px; margin-left: 10px",
                hx_post="/update", hx_target="#results",
                hx_trigger="change, input delay:500ms",
                hx_include="form",
                oninput=f"document.getElementById('{name}-slider').value = this.value"
            ),
            cls="flex items-center"
        ),
        style="margin-bottom: 0.75rem;"
    )


@matplotlib2fasthtml
def create_traffic_plot(simulator: LLMTrafficSimulator, traffic: np.ndarray, plot_type: str = 'traffic'):
    def plot_distribution(data, color='skyblue', line_color='r', label=None):
        plt.hist(data, bins=50, density=True, alpha=0.7, color=color)
        kde = gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 200)
        plt.plot(x_range, kde(x_range), f'{line_color}-', lw=2, label=label)
        
    def add_vertical_line(x, color, style, label, alpha=0.7):
        plt.axvline(x=x, color=color, linestyle=style, alpha=alpha, label=label)

    # Setup plot
    plt.figure(figsize=(12, 6))
    plt.xlabel('Requests per Second')
    plt.ylabel('Density')
    
    if plot_type == 'traffic':
        plt.title('User Traffic Distribution')
        plot_distribution(traffic, label='User Traffic Distribution')
        add_vertical_line(
            simulator.mean_turns_per_second, 
            'g', '--', 
            f'Mean: {simulator.mean_turns_per_second:.2f}'
        )
        
    elif plot_type == 'llm':
        plt.title('LLM Traffic Distribution')
        llm_traffic = traffic * simulator.config.llm_calls_per_turn
        plot_distribution(llm_traffic, label='LLM Traffic Distribution')
        
        # Add vertical lines for means and limits
        add_vertical_line(
            simulator.mean_requests_per_second,
            'g', '--',
            f'Mean: {simulator.mean_requests_per_second:.2f}'
        )
        add_vertical_line(
            simulator.max_requests_per_second,
            'orange', '-',
            f'Request Rate Limit: {simulator.max_requests_per_second:.2f}'
        )
        
    elif plot_type == 'guardrails':
        plt.title('Guardrails Text Units Distribution (per Second)')
        
        # Input distribution
        input_traffic = traffic * simulator.config.input_text_units
        input_mean = simulator.mean_turns_per_second * simulator.config.input_text_units
        input_quota = simulator.config.guardrails_request_quota

        # Output distribution
        output_traffic = traffic * simulator.config.output_text_units
        output_mean = simulator.mean_turns_per_second * simulator.config.output_text_units
        output_quota = simulator.config.contextual_grounding_quota
        
        # Calculate x-range that covers all data
        x_min = min(np.min(input_traffic), np.min(output_traffic), input_mean, output_mean, input_quota, output_quota)
        x_max = max(np.max(input_traffic), np.max(output_traffic), input_mean, output_mean, input_quota, output_quota)
        x_range = np.linspace(x_min, x_max, 200)
        
        # Plot input distribution
        kde_input = gaussian_kde(input_traffic)
        input_density = kde_input(x_range)
        plt.plot(x_range, input_density, color='#8B5CF6', alpha=0.7, label='Input Text Units')
        plt.fill_between(x_range, input_density, alpha=0.2, color='#8B5CF6')
        
        add_vertical_line(input_mean, '#6D28D9', '--', f'Input Mean: {input_mean:.2f}')
        add_vertical_line(input_quota, '#4C1D95', '-', 
                         f'Input Quota: {input_quota}')

        # Plot output distribution
        kde_output = gaussian_kde(output_traffic)
        output_density = kde_output(x_range)
        plt.plot(x_range, output_density, color='#2DD4BF', alpha=0.7, label='Output Text Units')
        plt.fill_between(x_range, output_density, alpha=0.2, color='#2DD4BF')
        
        add_vertical_line(output_mean, '#0D9488', '--', f'Output Mean: {output_mean:.2f}')
        add_vertical_line(output_quota, '#134E4A', '-',
                         f'Output Quota: {output_quota}')

        # Ensure y-axis starts at 0 and includes all data
        plt.ylim(bottom=0)
        
        # Add some padding to x-axis
        x_padding = (x_max - x_min) * 0.05
        plt.xlim(x_min - x_padding, x_max + x_padding)
        
        # Adjust legend position and style
        plt.legend(loc='upper right')
    
    return plt


def create_guardrail_toggles():
    def toggle_button(name: str, enabled: bool):
        enabled_cls = "bg-green-500 hover:bg-green-600" if enabled else "bg-gray-500 hover:bg-gray-600"
        status_text = "Enabled" if enabled else "Disabled"
        
        return Button(
            f"{name.replace('_', ' ').title()}: {status_text}",
            cls=f"{enabled_cls} text-white font-bold py-2 px-4 rounded w-full text-left",
            name=f"{name}_enabled",
            value=str(not enabled),  # Toggle value when clicked
            hx_post="/update-toggle",
            hx_target="#guardrail-toggles",
            hx_swap="outerHTML"
        )
    
    return Div(cls="space-y-4 mb-6", id="guardrail-toggles")(
        H3("Enable/Disable Guardrails", cls="text-lg font-bold"),
        Div(cls="space-y-2")(
            toggle_button("content_filter", state.config.content_filter_enabled),
            toggle_button("denied_topics", state.config.denied_topics_enabled),
            toggle_button("pii_filter", state.config.pii_filter_enabled),
            toggle_button("contextual_grounding", state.config.contextual_grounding_enabled),
        )
    )

def create_metrics_grid(analysis: dict, metrics_type: str = 'traffic'):
    metrics = []
    
    # Always show weekly requests
    metrics.append(
        Card(P(f"{analysis['weekly_requests']:,.0f}", cls="text-2xl font-bold"), 
            header=H3("Weekly Requests", cls="text-base")),
    )
    
    if metrics_type == 'llm':
        metrics.extend([
            Card(P(f"{analysis['weekly_llm_requests']:,.0f}", cls="text-2xl font-bold"), 
                header=H3("Weekly LLM Requests", cls="text-base")),
            Card(P(f"${analysis['weekly_llm_cost']:,.2f}", cls="text-2xl font-bold"), 
                 header=H3("Weekly LLM Cost", cls="text-base")),
            Card(P(f"${analysis['monthly_llm_cost']:,.2f}", cls="text-2xl font-bold"), 
                 header=H3("Monthly LLM Cost", cls="text-base")),
            Card(P(f"${analysis['llm_cost_per_conversation']:.3f}", cls="text-2xl font-bold"), 
                 header=H3("LLM Cost per Conversation", cls="text-base")),
        ])
    elif metrics_type == 'guardrails':
        metrics.extend([
            Card(P(f"{analysis['weekly_requests']:,.0f}", cls="text-2xl font-bold"), 
                header=H3("Weekly Guardrails Requests", cls="text-base")),
            Card(P(f"${analysis['weekly_guardrails_cost']:,.2f}", cls="text-2xl font-bold"), 
                 header=H3("Weekly Guardrails Cost", cls="text-base")),
            Card(P(f"${analysis['monthly_guardrails_cost']:,.2f}", cls="text-2xl font-bold"), 
                 header=H3("Monthly Guardrails Cost", cls="text-base")),
            Card(P(f"${analysis['guardrails_cost_per_conversation']:.3f}", cls="text-2xl font-bold"), 
                 header=H3("Guardrails Cost per Conversation", cls="text-base")),
        ])
        
        # Add text units and cost breakdown
        metrics.append(
            Card(
                Div(cls="space-y-2")(
                    *[P(f"{name.replace('_', ' ').title()}: {units:,.0f} text units", 
                        cls="text-sm") 
                      for name, units in analysis['text_units'].items()]
                ),
                header=H3("Text Units per Week", cls="text-base")
            )
        )
        
        # Add breakdown of guardrails costs
        for guardrail, cost in analysis['guardrails_breakdown'].items():
            if getattr(state.config, f"{guardrail}_enabled", True):
                metrics.append(
                    Card(P(f"${cost:,.2f}", cls="text-2xl font-bold"), 
                         header=H3(f"{guardrail.replace('_', ' ').title()} Cost", cls="text-base"))
                )
    
    return Grid(*metrics, columns="2", cls="gap-4 mb-4")


def create_tab_controls(tab_type: str):
    TRAFFIC_CONTROLS = [
        ("active_weekly_customers", 1, 10e6, state.config.active_weekly_customers, 1),
        ("customer_service_percentage", 1, 100, state.config.customer_service_percentage, 1),
        ("staged_rollout_percentage", 1, 100, state.config.staged_rollout_percentage, 1),
        ("distribution_variance", 0.01, 0.5, state.config.distribution_variance, 0.01),
        ("distribution_samples", 1000, 50000, state.config.distribution_samples, 1000),
    ]
    
    LLM_CONTROLS = [
        ("turns_per_conversation", 1, 20, state.config.turns_per_conversation, 1),
        ("llm_calls_per_turn", 1, 10, state.config.llm_calls_per_turn, 1),
        ("input_tokens_per_message", 50, 1000, state.config.input_tokens_per_message, 10),
        ("output_tokens_per_message", 10, 1000, state.config.output_tokens_per_message, 10),
        ("input_cost_per_1k", 0.001, 0.01, state.config.input_cost_per_1k, 0.001),
        ("output_cost_per_1k", 0.001, 0.05, state.config.output_cost_per_1k, 0.001),
        ("tokens_per_minute_limit", 50000, 1000000, state.config.tokens_per_minute_limit, 10000),
        ("requests_per_minute_limit", 1, 300, state.config.requests_per_minute_limit, 20),
    ]
    
    GUARDRAILS_CONTROLS = [
        ("input_text_units", 1, 20, state.config.input_text_units, 1),
        ("output_text_units", 1, 20, state.config.output_text_units, 1),
        ("content_filter_quota", 1, 100, state.config.content_filter_quota, 1),
        ("denied_topics_quota", 1, 100, state.config.denied_topics_quota, 1),
        ("pii_filter_quota", 1, 100, state.config.pii_filter_quota, 1),
        ("contextual_grounding_quota", 1, 200, state.config.contextual_grounding_quota, 1),
        ("guardrails_request_quota", 1, 100, state.config.guardrails_request_quota, 1),
    ]
    
    controls = TRAFFIC_CONTROLS if tab_type == 'traffic' else LLM_CONTROLS if tab_type == 'llm' else GUARDRAILS_CONTROLS if tab_type == 'guardrails'  else []
    return Form(*[create_control_input(*params) for params in controls])

# FastHTML App Setup
app, rt = fast_app(hdrs=(
    franken.Theme.orange.headers(),
    Script(defer=True, src="https://cdn.tailwindcss.com")
))

def run_simulation():
    state.simulator = LLMTrafficSimulator(state.config)
    state.traffic = state.simulator.generate_traffic()
    state.analysis = state.simulator.analyze_traffic(state.traffic)
     
if not state.simulator:
    run_simulation()

def get_results_content(tab_type: str = 'traffic'):
    """Generate just the results content for a given tab"""
    return Div(
        create_metrics_grid(state.analysis, tab_type),
        Div(create_traffic_plot(state.simulator, state.traffic, tab_type)),
        id="results"
    )

def traffic_tab():
    return Card(
        create_tab_controls('traffic'),
        get_results_content('traffic'),
        header=H2("Traffic Controls", cls="card-title"),
        cls="not-prose w-full"
    )

def llm_tab():
    return Card(
        create_tab_controls('llm'),
        get_results_content('llm'),
        header=H2("LLM Controls", cls="card-title"),
        cls="not-prose w-full"
    )

def guardrails_tab():
    return Card(
        create_guardrail_toggles(),
        create_tab_controls('guardrails'),
        Div(
            create_metrics_grid(state.analysis, 'guardrails'),
            Div(create_traffic_plot(state.simulator, state.traffic, 'guardrails')),
            id="results"
        ),
        header=H2("Guardrails Controls", cls="card-title"),
        cls="not-prose w-full"
    )

@rt("/update-toggle")
async def post(request):
    form = await request.form()
    
    # Update toggle state
    for field, value in form.items():
        if hasattr(state.config, field):
            setattr(state.config, field, value.lower() == 'true')
    
    run_simulation()
    
    # Return both the new toggles and the updated results
    return (
        create_guardrail_toggles(),
        Div(
            create_metrics_grid(state.analysis, 'guardrails'),
            Div(create_traffic_plot(state.simulator, state.traffic, 'guardrails')),
            id="results",
            hx_swap_oob="true"
        )
    )

@rt("/update")
async def post(request):
    form = await request.form()
    
    # Update numeric fields
    for field, value in form.items():
        if hasattr(state.config, field):
            if field.endswith('_enabled'):
                setattr(state.config, field, value.lower() == 'true')
            else:
                setattr(state.config, field, float(value))
    
    run_simulation()
    
    # Determine which tab to update based on the changed fields
    if any(field in form for field in [
        'active_weekly_customers', 'customer_service_percentage', 
        'staged_rollout_percentage', 'distribution_variance', 'distribution_samples'
    ]):
        return get_results_content('traffic')
    elif any(field in form for field in [
        'turns_per_conversation', 'llm_calls_per_turn', 'input_tokens_per_message',
        'output_tokens_per_message', 'input_cost_per_1k', 'output_cost_per_1k',
        'tokens_per_minute_limit', 'requests_per_minute_limit'
    ]):
        return get_results_content('llm')
    else:
        return get_results_content('guardrails')

# Route Handlers
@rt("/")
def get():
    if not state.simulator:
        run_simulation()
        
    content = Container(
        Card(
            P("Simulate application traffic patterns and service costs/quotas with Amazon Bedrock."),
            header=H1("Amazon Bedrock App Traffic Simulator", cls="text-4xl text-gray-200 mt-1"),
            footer=Button("Reset to Defaults", cls="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded",
                         hx_get="/reset", hx_target="body"),
            cls="not-prose w-full mb-4"
        ),
        Div(
            Div(cls="flex border-b border-gray-200")(
                A("Traffic", cls="px-4 py-2 hover:bg-gray-100", 
                  hx_get="/traffic", hx_target="#tab-content"),
                A("LLM", cls="px-4 py-2 hover:bg-gray-100",
                  hx_get="/llm", hx_target="#tab-content"),
                A("Guardrails", cls="px-4 py-2 hover:bg-gray-100",
                  hx_get="/guardrails", hx_target="#tab-content"),
            ),
            Div(traffic_tab(), id="tab-content", cls="mt-4")
        )
    )
    
    return Title("LLM Traffic Simulator"), content

@rt("/traffic")
def get():
    return traffic_tab()

@rt("/llm")
def get():
    return llm_tab()

@rt("/guardrails")
def get():
    return guardrails_tab()

@rt("/reset")
def reset():
    state.config = SimConfig()
    run_simulation()
    return RedirectResponse("/", status_code=303)

serve()