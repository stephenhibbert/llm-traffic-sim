
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
    staged_rollout_percentage: int = 1
    distribution_variance: float = 0.15
    distribution_samples: int = 10000
    
    # LLM Parameters
    human_turns_per_conversation: int = 5
    calls_per_turn: int = 3
    input_tokens_per_message: int = 100
    output_tokens_per_message: int = 20
    input_cost_per_1k: float = 0.003
    output_cost_per_1k: float = 0.015
    tokens_per_minute_limit: int = 200000
    invoke_model_requests_per_minute_limit: int = 20
    
    # Guardrail Parameters
    input_guardrail_text_units: int = 1  # Text units per input (PII/topics/content) guardrail message (1 unit = 1000 chars)
    output_guardrail_text_units: int = 5  # Text units per output (contextual grounding) guardrail message (1 unit = 1000 chars)
    content_filter_enabled: int = 1
    denied_topics_enabled: int = 0
    pii_filter_enabled: int = 0
    contextual_grounding_enabled: int = 1
    content_filter_price_per_1k: float = 0.75
    denied_topics_price_per_1k: float = 1.0
    contextual_grounding_price_per_1k: float = 0.1
    pii_filter_price_per_1k: float = 0.1    
    
    # Guardrails limits (text units per second)
    input_guardrail_limit: float = 25
    contextual_grounding_limit: float = 53
    
    # Guardrails request limit
    apply_guardrail_per_second_limit: float = 25


class SessionState:
    def __init__(self, session):
        self.session = session
        self._load_state()
    
    def _load_state(self):
        if 'config' not in self.session:
            self.config = SimConfig()
        else:
            self.config = SimConfig()
            # Convert stored values to correct types
            stored_config = self.session['config']
            for key, value in stored_config.items():
                if hasattr(self.config, key):
                    field_type = type(getattr(self.config, key))
                    try:
                        setattr(self.config, key, field_type(value))
                    except (ValueError, TypeError):
                        # Fallback to default if conversion fails
                        continue
        
        self.simulator = None
        self.traffic = None 
        self.analysis = None
    
    def save(self):
        # Store config as a dict with explicit type conversion
        self.session['config'] = {
            key: str(value) if isinstance(value, (int, float, bool)) else value
            for key, value in self.config.__dict__.items()
        }
        
    def reset(self):
        self.config = SimConfig()
        self.save()

def get_state(session):
    state = SessionState(session)
    run_simulation(state)
    return state

def update_state(_, state):
    state.save()

class LLMTrafficSimulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self._calculate_base_metrics()
    
    def _calculate_base_metrics(self):
        # Existing metrics calculation...
        weekly_cs_contacts = (self.config.active_weekly_customers * 
                            (self.config.customer_service_percentage / 100))
        self.conversations_per_week = weekly_cs_contacts * (self.config.staged_rollout_percentage / 100)
        
        total_turns = self.conversations_per_week * self.config.human_turns_per_conversation
        total_requests = total_turns * self.config.calls_per_turn
        self.mean_requests_per_second = total_requests / (7 * 24 * 60 * 60)
        self.mean_turns_per_second = total_turns / (7 * 24 * 60 * 60)
        
        # LLM limits
        self.max_requests_per_second = self.config.invoke_model_requests_per_minute_limit / 60
        self.max_input_tokens_per_second = self.config.tokens_per_minute_limit / 60
        
        
        self.text_units_per_request = {
            'content_filter': self.config.input_guardrail_text_units,
            'denied_topics': self.config.input_guardrail_text_units,
            'pii_filter': self.config.input_guardrail_text_units,
            'contextual_grounding': self.config.output_guardrail_text_units
        }
            
        # Store limits for plotting
        self.guardrails_limits = {
            'input_guardrail_limit': self.config.input_guardrail_limit,
            'contextual_grounding': self.config.contextual_grounding_limit,
            'requests': self.config.apply_guardrail_per_second_limit
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
            if getattr(self.config, f"{guardrail}_enabled", 1):
                text_units[guardrail] = requests_per_week * max(text_units_per_request, 1)
        return text_units

    def _calculate_guardrails_costs(self, requests_per_week):
        # Pricing per 1000 text units
        prices = {
            'content_filter': self.config.content_filter_price_per_1k,
            'denied_topics': self.config.denied_topics_price_per_1k,
            'contextual_grounding': self.config.contextual_grounding_price_per_1k,
            'pii_filter': self.config.pii_filter_price_per_1k
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
        weekly_llm_requests = weekly_requests * self.config.calls_per_turn
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
        'human_turns_per_conversation': 'Average number of back-and-forth messages in each conversation',
        'distribution_variance': 'Controls how much traffic varies from the mean (higher = more variance)',
        'distribution_samples': 'Number of data points to simulate (higher = smoother distribution)',
        
        # LLM Parameters
        'calls_per_turn': 'LLM API calls needed per assistant response (1 for simple completion, more for agents)',
        'input_tokens_per_message': 'Average number of tokens in each user message plus system prompts',
        'output_tokens_per_message': 'Average number of tokens in each assistant response',
        'input_cost_per_1k': 'Cost in USD per 1000 input tokens',
        'output_cost_per_1k': 'Cost in USD per 1000 output tokens',
        'tokens_per_minute_limit': 'Maximum tokens allowed per minute by the API',
        'invoke_model_requests_per_minute_limit': 'Maximum requests allowed per minute by the API',
        
        # Guardrail Parameters
        'input_guardrail_limit': 'Maximum PII/topics/content filter input text units processed per second (1000 chars = 1 text unit)',
        'contextual_grounding_limit': 'Maximum contextual grounding text units processed per second (1000 chars = 1 text unit)',
        'apply_guardrail_per_second_limit': 'Maximum guardrails API requests allowed per second',
        'input_guardrail_text_units': 'Number of text units per input message (1 text unit = 1000 characters)',
        'output_guardrail_text_units': 'Number of text units per output message (1 text unit = 1000 characters)',
        'content_filter_enabled': 'Content filter enabled (0 = no, 1 = yes)',
        'denied_topics_enabled': 'Denied topics enabled (0 = no, 1 = yes)',
        'pii_filter_enabled': 'PII filter enabled (0 = no, 1 = yes)',
        'contextual_grounding_enabled': 'Contextual grounding enabled (0 = no, 1 = yes)',
        
        # New Guardrail Price Parameters
        'content_filter_price_per_1k': 'Cost in USD per 1000 text units for content filtering',
        'denied_topics_price_per_1k': 'Cost in USD per 1000 text units for denied topics detection',
        'contextual_grounding_price_per_1k': 'Cost in USD per 1000 text units for contextual grounding',
        'pii_filter_price_per_1k': 'Cost in USD per 1000 text units for PII filtering'
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
                hx_indicator="body",
                hx_trigger="change, input delay:500ms",
                hx_include="form",
                oninput=f"document.getElementById('{name}-number').value = this.value"
            ),
            Input(
                type="number", id=f"{name}-number",
                min=str(min_val), max=str(max_val), value=str(default_val), step=str(step),
                style="width: 200px; margin-left: 10px",
                hx_post="/update", hx_target="#results",
                hx_trigger="change, input delay:500ms",
                hx_indicator="body",
                hx_include="form",
                oninput=f"document.getElementById('{name}-slider').value = this.value"
            ),
            cls="flex items-center"
        ),
        style="margin-bottom: 0.75rem;"
    )

@matplotlib2fasthtml
def create_traffic_plot(simulator: LLMTrafficSimulator, traffic: np.ndarray, plot_type: str = 'traffic'):
    def plot_distribution(data, kde, color='skyblue', line_color='r', label=None, alpha=0.7, fill_alpha=0.2, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.hist(data, bins=50, density=True, alpha=0.7, color=color)
        x_range = np.linspace(min(data), max(data), 200)
        ax.plot(x_range, kde(x_range), color=line_color, lw=2, alpha=alpha, label=label)
        ax.fill_between(x_range, kde(x_range), alpha=fill_alpha, color=color)
        
    def add_vertical_line(x, color, style, label, alpha=0.7, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.axvline(x=x, color=color, linestyle=style, alpha=alpha, label=label)
        
    def calculate_percent_over_limit(data, limit):
        return (np.sum(data > limit) / len(data)) * 100

    # Initialize figure to handle default case
    fig = None

    try:
        if plot_type == 'traffic':
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title('User Traffic Distribution')
            ax.set_xlabel('Requests per Second')
            ax.set_ylabel('Density')
            
            kde = gaussian_kde(traffic)
            plot_distribution(traffic, kde, label='User Traffic Distribution', ax=ax)
            add_vertical_line(
                simulator.mean_turns_per_second, 
                'g', '--', 
                f'Mean: {simulator.mean_turns_per_second:.2f}',
                ax=ax
            )
            
            ax.set_ylim(bottom=0)
            ax.legend(loc='upper right', framealpha=0.9)
            
        elif plot_type == 'llm':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
            fig.suptitle('LLM Traffic Distribution')

            llm_traffic = traffic * simulator.config.calls_per_turn
            input_token_traffic = llm_traffic * simulator.config.input_tokens_per_message

            # Top plot - Request Distribution
            ax1.set_xlabel('Requests per Second')
            ax1.set_ylabel('Request Density')

            kde_llm = gaussian_kde(llm_traffic)
            x_range = np.linspace(min(llm_traffic), max(llm_traffic), 200)

            ax1.hist(llm_traffic, bins=50, density=True, alpha=0.7, color='skyblue')
            ax1.plot(x_range, kde_llm(x_range), color='r', lw=2, label='Requests Traffic Distribution')

            # Calculate and display percentage over limit
            pct_over_req = calculate_percent_over_limit(llm_traffic, simulator.max_requests_per_second)
            ax1.text(0.02, 0.98, f'Requests over limit: {pct_over_req:.1f}%',
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8))

            ax1.axvline(x=simulator.mean_requests_per_second, color='g', linestyle='--',
                        label=f'Request Mean: {simulator.mean_requests_per_second:.2f}')
            ax1.axvline(x=simulator.max_requests_per_second, color='orange', linestyle='-',
                        label=f'Request Rate Limit: {simulator.max_requests_per_second:.2f}')

            ax1.set_ylim(bottom=0)
            ax1.legend(loc='upper right', framealpha=0.9)

            # Bottom plot - Input Token Distribution
            ax2.set_xlabel('Input Tokens per Second')
            ax2.set_ylabel('Input Token Density')

            kde_tokens = gaussian_kde(input_token_traffic)
            token_range = np.linspace(min(input_token_traffic), max(input_token_traffic), 200)

            ax2.hist(input_token_traffic, bins=50, density=True, alpha=0.7, color='lavender')
            ax2.plot(token_range, kde_tokens(token_range), color='purple', lw=2, 
                    label='Input Token Traffic Distribution')

            # Calculate and display percentage over limit
            pct_over_token = calculate_percent_over_limit(input_token_traffic, simulator.max_input_tokens_per_second)
            ax2.text(0.02, 0.98, f'Input tokens over limit: {pct_over_token:.1f}%',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8))

            input_token_mean = simulator.mean_requests_per_second * simulator.config.input_tokens_per_message
            
            ax2.axvline(x=input_token_mean, color='purple', linestyle='--',
                        label=f'Input Token Mean: {input_token_mean:.0f}')
            ax2.axvline(x=simulator.max_input_tokens_per_second, color='pink', linestyle='-',
                        label=f'Input Token Rate Limit: {simulator.max_input_tokens_per_second:.0f}')

            ax2.set_ylim(bottom=0)
            ax2.legend(loc='upper right', framealpha=0.9)

            plt.tight_layout()
            
        elif plot_type == 'guardrails':
            # Create two subplots: one for text units, one for requests
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Calculate traffic for text units
            input_traffic = np.zeros_like(traffic)
            if simulator.config.content_filter_enabled:
                input_traffic += traffic * simulator.config.input_guardrail_text_units
            if simulator.config.denied_topics_enabled:
                input_traffic += traffic * simulator.config.input_guardrail_text_units  
            if simulator.config.pii_filter_enabled:
                input_traffic += traffic * simulator.config.input_guardrail_text_units

            output_traffic = np.zeros_like(traffic)
            if simulator.config.contextual_grounding_enabled:
                output_traffic = traffic * simulator.config.output_guardrail_text_units
            total_traffic = input_traffic + output_traffic
            
            # Text Units Distribution (top plot)
            ax1.set_title('Text Units Distribution')
            ax1.set_xlabel('Text Units per Second')
            ax1.set_ylabel('Density')
            
            # Only calculate KDE if we have non-zero data
            if np.any(input_traffic):
                kde_input = gaussian_kde(input_traffic)
            if np.any(output_traffic):
                kde_output = gaussian_kde(output_traffic)
            
            input_mean = simulator.mean_turns_per_second * simulator.config.input_guardrail_text_units
            output_mean = simulator.mean_turns_per_second * simulator.config.output_guardrail_text_units
            
            input_limit = simulator.config.input_guardrail_limit
            output_limit = simulator.config.contextual_grounding_limit
            
            # Calculate percentages over limits for text units if we have data
            if np.any(input_traffic):
                pct_over_input = calculate_percent_over_limit(input_traffic, input_limit)
                ax1.text(0.02, 0.93, f'Input over limit: {pct_over_input:.1f}%',
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8))
                        
            if np.any(output_traffic):
                pct_over_output = calculate_percent_over_limit(output_traffic, output_limit)
                ax1.text(0.02, 0.88, f'Output over limit: {pct_over_output:.1f}%',
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(facecolor='white', alpha=0.8))
            
            # Plot distributions only if we have data
            if np.any(input_traffic):
                ax1.hist(input_traffic, bins=50, density=True, alpha=0.7, color='#8B5CF6')
                x_range = np.linspace(min(input_traffic), max(input_traffic), 200)
                ax1.plot(x_range, kde_input(x_range), color='#6D28D9', lw=2, label='Input Text Units')
                ax1.axvline(x=input_mean, color='#6D28D9', linestyle='--', 
                          label=f'Input Mean: {input_mean:.2f}')
                ax1.axvline(x=input_limit, color='#4C1D95', linestyle='-',
                          label=f'Input Limit: {input_limit}')
                
            if np.any(output_traffic):
                ax1.hist(output_traffic, bins=50, density=True, alpha=0.7, color='#2DD4BF')
                x_range = np.linspace(min(output_traffic), max(output_traffic), 200)
                ax1.plot(x_range, kde_output(x_range), color='#0D9488', lw=2, label='Output Text Units')
                ax1.axvline(x=output_mean, color='#0D9488', linestyle='--',
                          label=f'Output Mean: {output_mean:.2f}')
                ax1.axvline(x=output_limit, color='#134E4A', linestyle='-',
                          label=f'Output Limit: {output_limit}')
                
            ax1.set_ylim(bottom=0)
            ax1.legend(loc='upper right', framealpha=0.9)
            
            # Requests Distribution (bottom plot)
            ax2.set_title('Requests Distribution')
            ax2.set_xlabel('Requests per Second')
            ax2.set_ylabel('Density')
            
            # Plot request distribution
            kde_requests = gaussian_kde(traffic)
            plot_distribution(traffic, kde_requests, color='#94A3B8', line_color='#64748B',
                            label='Request Distribution', ax=ax2)
            
            request_mean = simulator.mean_turns_per_second
            request_limit = simulator.config.apply_guardrail_per_second_limit
            
            # Calculate percentage over limit for requests
            pct_over_requests = calculate_percent_over_limit(traffic, request_limit)
            
            # Add text for percentage on requests plot
            ax2.text(0.02, 0.98, f'Requests over limit: {pct_over_requests:.1f}%',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            # Add mean and limit lines for requests
            add_vertical_line(request_mean, '#64748B', '--', 
                            f'Request Mean: {request_mean:.2f}', ax=ax2)
            add_vertical_line(request_limit, '#475569', '-',
                            f'Request Limit: {request_limit}', ax=ax2)
            
            ax2.set_ylim(bottom=0)
            ax2.legend(loc='upper right', framealpha=0.9)
            
            plt.tight_layout()
        else:
            # Handle unknown plot type
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, f'Unknown plot type: {plot_type}', ha='center', va='center')
            
    except Exception as e:
        # Create an error figure if something goes wrong
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Error creating plot: {str(e)}', ha='center', va='center')
        
    return fig

def create_metrics_grid(analysis: dict, metrics_type: str = 'traffic', config=None):
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
        if config:
            for guardrail, cost in analysis['guardrails_breakdown'].items():
                if getattr(config, f"{guardrail}_enabled", 1):
                    metrics.append(
                        Card(P(f"${cost:,.2f}", cls="text-2xl font-bold"), 
                             header=H3(f"{guardrail.replace('_', ' ').title()} Cost", cls="text-base"))
                    )
    
    return Grid(*metrics, columns="2", cls="gap-4 mb-4")


def create_tab_controls(state, tab_type: str):
    TRAFFIC_CONTROLS = [
        ("active_weekly_customers", 1, 10e8, state.config.active_weekly_customers, 1),
        ("customer_service_percentage", 1, 100, state.config.customer_service_percentage, 1),
        ("staged_rollout_percentage", 1, 100, state.config.staged_rollout_percentage, 1),
        ("distribution_variance", 0.01, 0.5, state.config.distribution_variance, 0.01),
        ("distribution_samples", 1000, 50000, state.config.distribution_samples, 1000),
    ]
    
    LLM_CONTROLS = [
        ("human_turns_per_conversation", 1, 20, state.config.human_turns_per_conversation, 1),
        ("calls_per_turn", 1, 10, state.config.calls_per_turn, 1),
        ("input_tokens_per_message", 20, 1000, state.config.input_tokens_per_message, 10),
        ("output_tokens_per_message", 100, 1000, state.config.output_tokens_per_message, 10),
        ("input_cost_per_1k", 0.001, 0.01, state.config.input_cost_per_1k, 0.001),
        ("output_cost_per_1k", 0.001, 0.05, state.config.output_cost_per_1k, 0.001),
        ("tokens_per_minute_limit", 50000, 1000000, state.config.tokens_per_minute_limit, 10000),
        ("invoke_model_requests_per_minute_limit", 1, 300, state.config.invoke_model_requests_per_minute_limit, 20),
    ]
    
    GUARDRAILS_CONTROLS = [
        ("input_guardrail_text_units", 1, 50, state.config.input_guardrail_text_units, 1),
        ("output_guardrail_text_units", 1, 110, state.config.output_guardrail_text_units, 1),
        ("input_guardrail_limit", 1, 100, state.config.input_guardrail_limit, 1),
        ("contextual_grounding_limit", 1, 200, state.config.contextual_grounding_limit, 1),
        ("apply_guardrail_per_second_limit", 1, 100, state.config.apply_guardrail_per_second_limit, 1),
        
        ("content_filter_enabled", 0, 1, state.config.content_filter_enabled, 1),
        ("denied_topics_enabled", 0, 1, state.config.denied_topics_enabled, 1),
        ("pii_filter_enabled", 0, 1, state.config.pii_filter_enabled, 1),
        ("contextual_grounding_enabled", 0, 1, state.config.contextual_grounding_enabled, 1),
        
        ("content_filter_price_per_1k", 0.01, 2.0, state.config.content_filter_price_per_1k, 0.01),
        ("denied_topics_price_per_1k", 0.01, 2.0, state.config.denied_topics_price_per_1k, 0.01), 
        ("contextual_grounding_price_per_1k", 0.01, 1.0, state.config.contextual_grounding_price_per_1k, 0.01),
        ("pii_filter_price_per_1k", 0.01, 1.0, state.config.pii_filter_price_per_1k, 0.01),
    ]
    
    controls = TRAFFIC_CONTROLS if tab_type == 'traffic' else LLM_CONTROLS if tab_type == 'llm' else GUARDRAILS_CONTROLS if tab_type == 'guardrails'  else []
    return Form(*[create_control_input(*params) for params in controls])

# FastHTML App Setup
app, rt = fast_app(hdrs=(
    franken.Theme.orange.headers(),
    Script(defer=True, src="https://cdn.tailwindcss.com"),
    Style("""
        .htmx-request #spinner {
            display: flex !important;
        }
        
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid #fff;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    """)
))

def run_simulation(state):
    state.simulator = LLMTrafficSimulator(state.config)
    state.traffic = state.simulator.generate_traffic()
    state.analysis = state.simulator.analyze_traffic(state.traffic)

def get_results_content(state, tab_type: str = 'traffic'):
    return Div(
        create_metrics_grid(state.analysis, tab_type, state.config),
        Div(create_traffic_plot(state.simulator, state.traffic, tab_type)),
        id="results"
    )
    

def traffic_tab(state):
    return Card(
        create_tab_controls(state, 'traffic'),
        get_results_content(state, 'traffic'),
        header=H2("Traffic Controls", cls="card-title"),
        cls="not-prose w-full"
    )

def llm_tab(state):
    return Card(
        create_tab_controls(state, 'llm'),
        get_results_content(state, 'llm'),
        header=H2("LLM Controls", cls="card-title"),
        cls="not-prose w-full"
    )

def guardrails_tab(state):
    return Card(
        Div(
            create_tab_controls(state, 'guardrails'),
            Div(
                create_metrics_grid(state.analysis, 'guardrails', state.config),
                Div(create_traffic_plot(state.simulator, state.traffic, 'guardrails')),
                id="results"
            ),
            id="guardrails-content"
        ),
        header=H2("Guardrails Controls", cls="card-title"),
        cls="not-prose w-full"
    )


@rt("/update") 
async def post(request, session):
    form = await request.form()
    state = get_state(session)
    
    # Update numeric fields and toggle fields
    for field, value in form.items():
        if hasattr(state.config, field):
            try:
                # Update numeric or other typed fields
                field_type = type(getattr(state.config, field))
                setattr(state.config, field, field_type(value))
            except (ValueError, TypeError):
                continue
    
    # Run the simulation after all form data has been processed
    run_simulation(state)
    state.save()

    # Determine the appropriate content to refresh
    if any(field in form for field in [
        'active_weekly_customers', 'customer_service_percentage', 
        'staged_rollout_percentage', 'distribution_variance', 'distribution_samples'
    ]):
        return get_results_content(state, 'traffic')
    elif any(field in form for field in [
        'human_turns_per_conversation', 'calls_per_turn', 'input_tokens_per_message',
        'output_tokens_per_message', 'input_cost_per_1k', 'output_cost_per_1k',
        'tokens_per_minute_limit', 'invoke_model_requests_per_minute_limit'
    ]):
        return get_results_content(state, 'llm')
    else:
        # If guardrail toggle changes, refresh guardrails content
        return get_results_content(state, 'guardrails')



# Route Handlers
@rt("/")
def get(session):

    if 'config' not in session:
        state = SessionState(session)
        run_simulation(state)
        update_state(session, state)
    else:
        state = get_state(session)
        
    content = Container(
            Card(
                P("Simulate application traffic patterns and service costs/limits with Amazon Bedrock."),
                P(
                    "Numbers correct on 31/10/2024 - please check here for latest: ",
                    A("AWS Bedrock Documentation", 
                    href="https://docs.aws.amazon.com/general/latest/gr/bedrock.html",
                    target="_blank",
                    rel="noopener noreferrer",
                    cls="text-blue-600 hover:text-blue-800 hover:underline transition-colors duration-200")
                ),
                P(
                    "Spotted a mistake? Please raise an issue on ",
                    A("Github",
                    href="https://github.com/stephenhibbert/llm-traffic-sim",
                    target="_blank",
                    rel="noopener noreferrer",
                    cls="text-blue-600 hover:text-blue-800 hover:underline transition-colors duration-200")
                ),
                header=H1("Amazon Bedrock App Traffic Simulator", 
                        cls="text-4xl text-gray-200 mt-1"),
                footer=Div(
                    Button("Reset to Defaults", 
                        cls="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded transition-colors duration-200",
                        hx_get="/reset", 
                        hx_target="body")
                ),
                cls="not-prose w-full mb-4"
            ),
                Div(
        # Semi-transparent dark backdrop with blur
        Div(cls="fixed inset-0 bg-gray-900/50 backdrop-blur-sm z-40 transition-opacity duration-200"),
        # Centered loading spinner and text
        Div(
            Div(cls="spinner"),
            P("Running simulation...", cls="text-white mt-4 text-xl"),
            cls="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-50 flex flex-col items-center"
        ),
        id="spinner",
        cls="htmx-indicator fixed inset-0 hidden"  # Hidden by default
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
            Div(traffic_tab(state), id="tab-content", cls="mt-4"),
        )
    )
    return content

@rt("/traffic")
def get(session):
    state = get_state(session)
    return traffic_tab(state)

@rt("/llm") 
def get(session):
    state = get_state(session)
    return llm_tab(state)

@rt("/guardrails")
def get(session):
    state = get_state(session)
    return guardrails_tab(state)

@rt("/reset")
def get(session):
    state = SessionState(session)
    state.reset()
    run_simulation(state)
    return RedirectResponse("/", status_code=303)

serve() 


