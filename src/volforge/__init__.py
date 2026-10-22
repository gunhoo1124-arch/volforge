"""VolForge quantitative derivatives research toolkit."""

from volforge.basket import (
    BasketAssetMarket,
    BasketMarket,
    BasketMonteCarloSettings,
    BasketPricingResult,
    parse_basket_market,
    price_basket_option,
)
from volforge.basket_covariance import (
    BasketCovarianceSelection,
    BasketCovarianceSelectionSettings,
    materialize_selected_covariance_replay,
    select_basket_covariance,
    verify_basket_covariance_selection,
)
from volforge.basket_estimation import (
    BasketEstimationSettings,
    BasketInputEstimate,
    estimate_basket_market,
    generate_synthetic_smile_basket_inputs,
    generate_synthetic_statistical_basket_inputs,
    materialize_estimated_replay,
    verify_basket_input_estimate,
)
from volforge.basket_inputs import (
    BasketInputSnapshot,
    BasketInputStore,
    BasketInputVerification,
    generate_synthetic_basket_inputs,
)
from volforge.basket_option_inputs import (
    BasketOptionQuoteSnapshot,
    BasketOptionQuoteStore,
    BasketOptionQuoteVerification,
    generate_synthetic_option_quote_inputs,
    verify_option_quote_snapshot,
)
from volforge.basket_replay import (
    BasketReplayResult,
    BasketReplaySettings,
    generate_synthetic_basket_replay,
    run_basket_hedge_replay,
    verify_basket_replay_hash_chain,
)
from volforge.basket_risk_hedging import (
    BasketRiskHedgeResult,
    BasketRiskHedgeSettings,
    run_basket_risk_hedge_replay,
    verify_basket_risk_hedge_hash_chain,
)
from volforge.deliverables import (
    DeliverableTerms,
    SettlementValuation,
    parse_deliverable_terms,
    value_option_settlement,
)
from volforge.implied_vol import ImpliedVolResult, implied_volatility
from volforge.models.common import Greeks, OptionType, PricingResult

__version__ = "1.11.0"

__all__ = [
    "BasketAssetMarket",
    "BasketCovarianceSelection",
    "BasketCovarianceSelectionSettings",
    "BasketEstimationSettings",
    "BasketInputEstimate",
    "BasketInputSnapshot",
    "BasketInputStore",
    "BasketInputVerification",
    "BasketMarket",
    "BasketMonteCarloSettings",
    "BasketOptionQuoteSnapshot",
    "BasketOptionQuoteStore",
    "BasketOptionQuoteVerification",
    "BasketPricingResult",
    "BasketReplayResult",
    "BasketReplaySettings",
    "BasketRiskHedgeResult",
    "BasketRiskHedgeSettings",
    "DeliverableTerms",
    "Greeks",
    "ImpliedVolResult",
    "OptionType",
    "PricingResult",
    "SettlementValuation",
    "estimate_basket_market",
    "generate_synthetic_basket_inputs",
    "generate_synthetic_basket_replay",
    "generate_synthetic_option_quote_inputs",
    "generate_synthetic_smile_basket_inputs",
    "generate_synthetic_statistical_basket_inputs",
    "implied_volatility",
    "materialize_estimated_replay",
    "materialize_selected_covariance_replay",
    "parse_basket_market",
    "parse_deliverable_terms",
    "price_basket_option",
    "run_basket_hedge_replay",
    "run_basket_risk_hedge_replay",
    "select_basket_covariance",
    "value_option_settlement",
    "verify_basket_covariance_selection",
    "verify_basket_input_estimate",
    "verify_basket_replay_hash_chain",
    "verify_basket_risk_hedge_hash_chain",
    "verify_option_quote_snapshot",
]
