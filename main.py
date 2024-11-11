
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
    input_tokens_per_message: int = 20
    output_tokens_per_message: int = 100
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
        self.traffic = self.generate_traffic_distribution(self.mean_conversations_per_second)

    def _calculate_base_metrics(self):
        """Calculate all core metrics used throughout the simulator"""
        # Base conversation metrics
        weekly_cs_contacts = (self.config.active_weekly_customers * 
                            (self.config.customer_service_percentage / 100))
        self.conversations_per_week = weekly_cs_contacts * (self.config.staged_rollout_percentage / 100)
        self.mean_conversations_per_second = self.conversations_per_week / (7 * 24 * 60 * 60)

        # LLM metrics
        self.llm_multiplier = self.config.human_turns_per_conversation * self.config.calls_per_turn
        self.mean_llm_requests_per_second = self.mean_conversations_per_second * self.llm_multiplier
        self.mean_input_tokens_per_second = self.mean_llm_requests_per_second * self.config.input_tokens_per_message
        
        # API Limits
        self.max_requests_per_second = self.config.invoke_model_requests_per_minute_limit / 60
        self.max_input_tokens_per_second = self.config.tokens_per_minute_limit / 60

        # Guardrails counts
        self.enabled_input_guardrails = sum([
            bool(self.config.content_filter_enabled),
            bool(self.config.denied_topics_enabled),
            bool(self.config.pii_filter_enabled)
        ])
        self.enabled_output_guardrails = bool(self.config.contextual_grounding_enabled)
        self.total_enabled_guardrails = self.enabled_input_guardrails + self.enabled_output_guardrails

        # Guardrails multipliers
        self.input_guardrail_multiplier = (self.config.human_turns_per_conversation * 
                                         self.config.input_guardrail_text_units * 
                                         self.enabled_input_guardrails)
        self.output_guardrail_multiplier = (self.config.human_turns_per_conversation * 
                                          self.config.output_guardrail_text_units * 
                                          self.enabled_output_guardrails)
        
        # Weekly metrics (for cost calculations)
        self.weekly_llm_requests = self.conversations_per_week * self.llm_multiplier
        self.weekly_guardrails_requests = self.conversations_per_week * self.config.human_turns_per_conversation * self.total_enabled_guardrails

    def _calculate_llm_costs(self):
        """Calculate LLM-related costs"""
        total_input_tokens = self.weekly_llm_requests * self.config.input_tokens_per_message
        total_output_tokens = self.weekly_llm_requests * self.config.output_tokens_per_message
        
        return ((total_input_tokens / 1000) * self.config.input_cost_per_1k +
                (total_output_tokens / 1000) * self.config.output_cost_per_1k)

    def _calculate_guardrails_costs(self):
        """Calculate costs for each guardrail service"""
        requests_per_guardrail = self.weekly_guardrails_requests / self.total_enabled_guardrails
        
        return {
            'content_filter': (requests_per_guardrail * self.config.input_guardrail_text_units / 1000 * 
                            self.config.content_filter_price_per_1k * self.config.content_filter_enabled),
            'denied_topics': (requests_per_guardrail * self.config.input_guardrail_text_units / 1000 * 
                            self.config.denied_topics_price_per_1k * self.config.denied_topics_enabled),
            'pii_filter': (requests_per_guardrail * self.config.input_guardrail_text_units / 1000 * 
                        self.config.pii_filter_price_per_1k * self.config.pii_filter_enabled),
            'contextual_grounding': (requests_per_guardrail * self.config.output_guardrail_text_units / 1000 * 
                                self.config.contextual_grounding_price_per_1k * self.config.contextual_grounding_enabled)
        }

    def _calculate_text_units(self):
        """Calculate weekly text units for each service"""
        return {
            name: (self.weekly_guardrails_requests / self.total_enabled_guardrails) * (
                self.config.input_guardrail_text_units if 'grounding' not in name 
                else self.config.output_guardrail_text_units
            ) * bool(enabled)
            for name, enabled in {
                'content_filter': self.config.content_filter_enabled,
                'denied_topics': self.config.denied_topics_enabled,
                'pii_filter': self.config.pii_filter_enabled,
                'contextual_grounding': self.config.contextual_grounding_enabled
            }.items()
        }

    import numpy as np

    def generate_traffic_distribution(self, base_traffic, multiplier=1):
        """Generate a realistic UK traffic distribution with daily and weekly patterns
        while maintaining the correct mean
        
        Args:
            base_traffic (float): Base traffic level
            multiplier (float): Overall traffic multiplier
            
        Returns:
            numpy.ndarray: Array of traffic values following UK patterns
        """
        scaled_traffic = base_traffic * multiplier
        samples = int(self.config.distribution_samples)
        
        # Generate base distribution
        traffic = np.random.normal(
            scaled_traffic,
            scaled_traffic * self.config.distribution_variance,
            samples
        )
        
        # Apply time-of-day factors (UK business hours)
        hour_factors = np.array([
            0.1, 0.1, 0.1, 0.1, 0.1, 0.2,  # 00-05: Very low night traffic
            0.3, 0.6, 0.9, 1.2, 1.3, 1.2,  # 06-11: Morning ramp-up
            1.1, 1.0, 1.1, 1.2, 1.3, 1.2,  # 12-17: Business hours peak
            1.0, 0.8, 0.6, 0.4, 0.2, 0.1   # 18-23: Evening wind-down
        ])
        
        # Normalize hourly factors to maintain mean
        hour_factors = hour_factors / np.mean(hour_factors)
        
        # Apply weekly pattern (UK work week)
        day_factors = np.array([
            0.7,   # Sunday
            1.0,   # Monday
            1.1,   # Tuesday
            1.1,   # Wednesday
            1.0,   # Thursday
            0.9,   # Friday
            0.6    # Saturday
        ])
        
        # Normalize daily factors to maintain mean
        day_factors = day_factors / np.mean(day_factors)
        
        # Apply hourly and daily patterns
        hours = np.arange(samples) % 24
        days = (np.arange(samples) // 24) % 7
        
        traffic = traffic * hour_factors[hours] * day_factors[days]
        
        return np.maximum(traffic, 0)  # Ensure no negative traffic
    
    def get_traffic_metrics(self, traffic_type='all'):
        """Get all traffic metrics for plotting"""
        metrics = {
            'conversation': {
                'distribution': self.generate_traffic_distribution(self.mean_conversations_per_second),
                'mean': self.mean_conversations_per_second,
                'limit': None
            },
            'llm_requests': {
                'distribution': self.generate_traffic_distribution(self.mean_conversations_per_second, self.llm_multiplier),
                'mean': self.mean_llm_requests_per_second,
                'limit': self.max_requests_per_second
            },
            'input_tokens': {
                'distribution': self.generate_traffic_distribution(self.mean_conversations_per_second, 
                    self.llm_multiplier * self.config.input_tokens_per_message),
                'mean': self.mean_input_tokens_per_second,
                'limit': self.max_input_tokens_per_second
            },
            'guardrails': {
                'input': {
                    'distribution': np.zeros(self.config.distribution_samples) if not self.enabled_input_guardrails else
                        self.generate_traffic_distribution(self.mean_conversations_per_second, self.input_guardrail_multiplier),
                    'mean': self.mean_conversations_per_second * self.input_guardrail_multiplier,
                    'limit': self.config.input_guardrail_limit
                },
                'output': {
                    'distribution': np.zeros(self.config.distribution_samples) if not self.enabled_output_guardrails else
                        self.generate_traffic_distribution(self.mean_conversations_per_second, self.output_guardrail_multiplier),
                    'mean': self.mean_conversations_per_second * self.output_guardrail_multiplier,
                    'limit': self.config.contextual_grounding_limit
                },
                'requests': {
                    'distribution': np.zeros(self.config.distribution_samples) if not self.total_enabled_guardrails else
                        self.generate_traffic_distribution(
                            self.mean_conversations_per_second,
                            self.config.human_turns_per_conversation * self.total_enabled_guardrails
                        ),
                    'mean': self.mean_conversations_per_second * self.config.human_turns_per_conversation * self.total_enabled_guardrails,
                    'limit': self.config.apply_guardrail_per_second_limit
                }
            }
        }

        if traffic_type != 'all':
            return metrics.get(traffic_type, {})
        return metrics

    def analyze_traffic(self):
        """Calculate costs and other metrics"""
        weekly_llm_cost = self._calculate_llm_costs()
        guardrails_costs = self._calculate_guardrails_costs()
        weekly_guardrails_cost = sum(guardrails_costs.values())

        return {
            'weekly_conversations': self.conversations_per_week,
            'weekly_llm_requests': self.weekly_llm_requests,
            'weekly_guardrails_requests': self.weekly_guardrails_requests,
            'weekly_llm_cost': weekly_llm_cost,
            'weekly_guardrails_cost': weekly_guardrails_cost,
            'monthly_llm_cost': weekly_llm_cost * 4.33,
            'monthly_guardrails_cost': weekly_guardrails_cost * 4.33,
            'llm_cost_per_conversation': weekly_llm_cost / self.conversations_per_week,
            'guardrails_cost_per_conversation': weekly_guardrails_cost / self.conversations_per_week,
            'guardrails_breakdown': guardrails_costs,
            'text_units': self._calculate_text_units()
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

def plot_multiple_distributions(metrics_list, ax, labels=None, title=None, xlabel=None):
    # Check if we have any non-zero distributions first
    has_data = False
    for metrics in metrics_list:
        if np.any(metrics['distribution']):
            has_data = True
            break
            
    if not has_data:
        ax.text(0.5, 0.5, "No traffic (all guardrails disabled)", 
                ha='center', va='center', transform=ax.transAxes)
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        return
    
    colors = ['#8B5CF6', '#2DD4BF']  # Purple and teal for input/output
    line_colors = ['#6D28D9', '#0D9488']
    
    for i, metrics in enumerate(metrics_list):
        if not np.any(metrics['distribution']):
            continue
            
        data = metrics['distribution']
        kde = gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 200)
        
        label = labels[i] if labels else f'Distribution {i+1}'
        
        # Plot histogram and KDE
        ax.hist(data, bins=150, density=True, alpha=0.7, color=colors[i])
        ax.plot(x_range, kde(x_range), color=line_colors[i], lw=2, 
                label=f'{label} Distribution')
        
        # Add mean line
        if metrics['mean'] is not None:
            ax.axvline(x=metrics['mean'], color=line_colors[i], linestyle='--',
                      label=f'{label} Mean: {metrics["mean"]:.2f}')
        
        # Add limit line
        if metrics['limit'] is not None:
            ax.axvline(x=metrics['limit'], color=line_colors[i], linestyle='-',
                      label=f'{label} Limit: {metrics["limit"]}')
            
            # Calculate and show percentage over limit
            pct_over = (np.sum(data > metrics['limit']) / len(data)) * 100
            y_pos = 0.98 - (i * 0.05)  # Stagger text vertically
            ax.text(0.02, y_pos, f'{label} over limit: {pct_over:.1f}%',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.8))
    
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', framealpha=0.9)

@matplotlib2fasthtml
def create_traffic_plot(simulator: LLMTrafficSimulator, plot_type: str = 'traffic'):
    fig = plt.figure(figsize=(12, 6))
    
    if plot_type == 'traffic':
        metrics = simulator.get_traffic_metrics('conversation')
        ax = fig.add_subplot(111)
        plot_distribution(metrics['distribution'], ax, 
                         mean=metrics['mean'],
                         title='User Traffic Distribution',
                         xlabel='Conversations per Second')
        
    elif plot_type == 'llm':
        metrics = simulator.get_traffic_metrics()
        fig.set_size_inches(12, 10)
        
        # Requests plot
        ax1 = fig.add_subplot(211)
        plot_distribution(metrics['llm_requests']['distribution'], ax1,
                         mean=metrics['llm_requests']['mean'],
                         limit=metrics['llm_requests']['limit'],
                         title='LLM Request Distribution',
                         xlabel='Requests per Second')
        
        # Tokens plot
        ax2 = fig.add_subplot(212)
        plot_distribution(metrics['input_tokens']['distribution'], ax2,
                         mean=metrics['input_tokens']['mean'],
                         limit=metrics['input_tokens']['limit'],
                         title='Input Token Distribution',
                         xlabel='Input Tokens per Second')
        
    elif plot_type == 'guardrails':
        metrics = simulator.get_traffic_metrics()
        fig.set_size_inches(12, 10)
        
        # Text units plot
        ax1 = fig.add_subplot(211)
        plot_multiple_distributions(
            [metrics['guardrails']['input'], metrics['guardrails']['output']],
            ax1, labels=['Input', 'Output'],
            title='Text Units Distribution',
            xlabel='Text Units per Second'
        )
        
        # Requests plot
        ax2 = fig.add_subplot(212)
        plot_distribution(
            metrics['guardrails']['requests']['distribution'],
            ax2,
            mean=metrics['guardrails']['requests']['mean'],
            limit=metrics['guardrails']['requests']['limit'],
            title='Guardrails Request Distribution',
            xlabel='Requests per Second'
        )
    
    plt.tight_layout()
    return fig

def plot_distribution(data, ax, mean=None, limit=None, title=None, xlabel=None):
    """Helper function to plot a single distribution with KDE"""
    kde = gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 200)
    
    ax.hist(data, bins=150, density=True, alpha=0.7, color='skyblue')
    ax.plot(x_range, kde(x_range), color='r', lw=2, label='Distribution')
    
    if mean is not None:
        ax.axvline(x=mean, color='g', linestyle='--', label=f'Mean: {mean:.2f}')
    if limit is not None:
        ax.axvline(x=limit, color='orange', linestyle='-', label=f'Limit: {limit:.2f}')
        pct_over = (np.sum(data > limit) / len(data)) * 100
        ax.text(0.02, 0.98, f'Over limit: {pct_over:.1f}%',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend(loc='upper right')

def create_metrics_grid(analysis: dict, metrics_type: str = 'traffic', config=None):
    metrics = []
    
    # Always show weekly requests
    metrics.append(
        Card(P(f"{analysis['weekly_conversations']:,.0f}", cls="text-2xl font-bold"), 
            header=H3("Weekly Conversations", cls="text-base")),
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
            Card(P(f"{analysis['weekly_guardrails_requests']:,.0f}", cls="text-2xl font-bold"), 
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
    state.analysis = state.simulator.analyze_traffic()

def get_results_content(state, tab_type: str = 'traffic'):
    return Div(
        create_metrics_grid(state.analysis, tab_type, state.config),
        Div(create_traffic_plot(state.simulator, tab_type)),
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
                Div(create_traffic_plot(state.simulator, 'guardrails')),
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


