//! Rust compute kernels for staged migration.
//!
//! `run_backtest_kernel` contains the quarterly backtest core behind
//! `kernel-backtest`. Ranking and constructor kernels now execute behind
//! their feature flags.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap, HashSet};

type JsonMap = Map<String, Value>;

const FLOAT_EPSILON: f64 = 1e-12;
const TARGET_DELTA_EPSILON: f64 = 1e-9;

const DEFAULT_COMMISSION_RATE: f64 = 0.0;
const DEFAULT_SLIPPAGE_BPS: f64 = 10.0;
const DEFAULT_FX_FEE_RATE: f64 = 0.0;
const DEFAULT_KR_SELL_TAX_RATE: f64 = 0.0015;
const DEFAULT_JP_REALIZED_GAINS_TAX_RATE: f64 = 0.20;
const DEFAULT_US_REALIZED_GAINS_TAX_RATE: f64 = 0.22;

const DEFAULT_MAX_HOLDINGS: usize = 20;
const DEFAULT_MAX_SINGLE_NAME_WEIGHT: f64 = 0.08;
const DEFAULT_COUNTRY_TOLERANCE: f64 = 0.02;
const DEFAULT_TE_ACTIVE_L2_CAP: f64 = 0.08;
const DEFAULT_ALPHA_TILT_STRENGTH: f64 = 0.35;
const DEFAULT_MAX_ADV_PARTICIPATION: f64 = 0.10;
const DEFAULT_PORTFOLIO_VALUE_KRW: f64 = 10_000_000.0;
const SENTINEL_RANK: i64 = 1_000_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelError {
    NotEnabled,
    NotImplemented,
    InvalidRequest,
    ExecutionFailed,
    SerializationFailed,
}

pub type KernelResult = Result<String, KernelError>;

#[derive(Debug, Deserialize)]
struct BacktestKernelRequest {
    rebalance_schedule: Vec<Value>,
    prices: Vec<JsonMap>,
    benchmark_series: Vec<JsonMap>,
    fx_rates: Vec<JsonMap>,
    portfolio_allocations: Value,
    initial_nav_krw: f64,
    #[serde(default)]
    cost_tax_config: Option<JsonMap>,
}

#[derive(Debug, Deserialize)]
struct RankingKernelRequest {
    factor_rows: Vec<JsonMap>,
    accepted_rows: Vec<JsonMap>,
    sector_by_security: HashMap<String, String>,
    requested_counts_by_country: HashMap<String, i64>,
    factor_model: String,
    sector_active_band: f64,
    use_size_stratification: bool,
}

#[derive(Debug, Clone, Serialize)]
struct RankingCandidate {
    security_id: String,
    country: String,
    factor_value: f64,
    base_factor_value: f64,
    sector: String,
    median_traded_value_krw: f64,
    rd_expense: f64,
    sales_ttm: Option<f64>,
    size_bucket: String,
    rank_in_country: Option<i64>,
    is_eligible: bool,
}

#[derive(Debug, Clone)]
struct CostTaxConfig {
    commission_rate: f64,
    slippage_rate: f64,
    fx_fee_rate: f64,
    kr_sell_tax_rate: f64,
    jp_realized_gains_tax_rate: f64,
    us_realized_gains_tax_rate: f64,
}

#[derive(Debug, Clone)]
struct Position {
    country: String,
    currency: String,
    quantity: f64,
    avg_cost_krw: f64,
}

#[derive(Debug, Clone)]
struct Allocation {
    security_id: String,
    target_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TradeLedgerEntry {
    rebalance_date: String,
    security_id: String,
    side: String,
    country: String,
    currency: String,
    quantity: f64,
    price_local: f64,
    price_krw: f64,
    notional_krw: f64,
    gross_proceeds_krw: f64,
    gross_cost_krw: f64,
    commission_krw: f64,
    slippage_krw: f64,
    fx_fee_krw: f64,
    sell_tax_krw: f64,
    realized_gain_krw: f64,
    realized_gains_tax_krw: f64,
    total_cost_tax_krw: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct HoldingSnapshot {
    as_of_date: String,
    security_id: String,
    country: String,
    currency: String,
    quantity: f64,
    avg_cost_krw: f64,
    price_local: f64,
    price_krw: f64,
    market_value_krw: f64,
    weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct PeriodReturn {
    start_date: String,
    end_date: String,
    start_nav_krw: f64,
    end_nav_krw: f64,
    gross_return: f64,
    net_return: f64,
    benchmark_krw_return: Option<f64>,
    total_commission_krw: f64,
    total_slippage_krw: f64,
    total_fx_fee_krw: f64,
    total_sell_tax_krw: f64,
    total_realized_gains_tax_krw: f64,
    total_cost_tax_krw: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BacktestMetrics {
    periods: usize,
    start_nav_krw: f64,
    end_nav_krw: f64,
    cumulative_net_return: f64,
    cumulative_gross_return: f64,
    benchmark_cumulative_return: Option<f64>,
    total_commission_krw: f64,
    total_slippage_krw: f64,
    total_fx_fee_krw: f64,
    total_sell_tax_krw: f64,
    total_realized_gains_tax_krw: f64,
    total_cost_tax_krw: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct BacktestResult {
    trades: Vec<TradeLedgerEntry>,
    holdings: Vec<HoldingSnapshot>,
    returns: Vec<PeriodReturn>,
    metrics: BacktestMetrics,
}

#[derive(Debug, Deserialize)]
struct ConstructorKernelRequest {
    ranked_factor_rows: Vec<JsonMap>,
    #[serde(default)]
    country_targets: Option<Vec<(String, f64)>>,
    #[serde(default)]
    risk_controls: Option<JsonMap>,
}

#[derive(Debug, Clone)]
struct ConstructorRankedRow {
    security_id: String,
    country: String,
    factor_value: f64,
    rank_in_country: Option<i64>,
    benchmark_proxy_weight: Option<f64>,
    median_traded_value_krw: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ConstructorHolding {
    security_id: String,
    country: String,
    weight: f64,
    rank_in_country: Option<i64>,
    factor_value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ConstructorSelectionDiagnostics {
    requested_holdings: usize,
    selected_holdings: usize,
    available_eligible: usize,
    requested_country_counts: Vec<(String, usize)>,
    available_country_counts: Vec<(String, usize)>,
    selected_country_counts: Vec<(String, usize)>,
    country_weights: Vec<(String, f64)>,
    cash_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ConstructorResult {
    holdings: Vec<ConstructorHolding>,
    fallback_triggered: bool,
    fallback_reasons: Vec<String>,
    jp_odd_lot_enabled: bool,
    diagnostics: ConstructorSelectionDiagnostics,
}

fn value_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.trim().to_string(),
        _ => value.to_string().trim().to_string(),
    }
}

fn value_to_upper_text(value: &Value) -> String {
    value_to_text(value).to_ascii_uppercase()
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

fn parse_iso_date_parts(value: &str) -> Option<(i32, u32, u32)> {
    if value.len() != 10 {
        return None;
    }
    let bytes = value.as_bytes();
    if bytes[4] != b'-' || bytes[7] != b'-' {
        return None;
    }
    let year = value[0..4].parse::<i32>().ok()?;
    let month = value[5..7].parse::<u32>().ok()?;
    let day = value[8..10].parse::<u32>().ok()?;
    if month == 0 || month > 12 {
        return None;
    }
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => return None,
    };
    if day == 0 || day > days_in_month {
        return None;
    }
    Some((year, month, day))
}

fn to_iso_date(value: &Value) -> Result<String, KernelError> {
    let text = value_to_text(value);
    if text.is_empty() || parse_iso_date_parts(&text).is_none() {
        return Err(KernelError::ExecutionFailed);
    }
    Ok(text)
}

fn to_float(value: &Value) -> Result<f64, KernelError> {
    let parsed = match value {
        Value::Number(number) => number.as_f64().ok_or(KernelError::ExecutionFailed)?,
        Value::String(text) => text
            .trim()
            .parse::<f64>()
            .map_err(|_| KernelError::ExecutionFailed)?,
        _ => value
            .to_string()
            .trim()
            .parse::<f64>()
            .map_err(|_| KernelError::ExecutionFailed)?,
    };
    if !parsed.is_finite() {
        return Err(KernelError::ExecutionFailed);
    }
    Ok(parsed)
}

fn normalize_rate(value: &Value) -> Result<f64, KernelError> {
    let parsed = to_float(value)?;
    if !(0.0..=1.0).contains(&parsed) {
        return Err(KernelError::ExecutionFailed);
    }
    Ok(parsed)
}

fn config_value<'a>(config: Option<&'a JsonMap>, key: &str) -> Option<&'a Value> {
    config.and_then(|row| row.get(key))
}

fn parse_cost_tax_config(config: Option<&JsonMap>) -> Result<CostTaxConfig, KernelError> {
    let commission_rate = match config_value(config, "commission_rate") {
        Some(value) => normalize_rate(value)?,
        None => DEFAULT_COMMISSION_RATE,
    };

    let slippage_bps = match config_value(config, "slippage_bps") {
        Some(value) => to_float(value)?,
        None => DEFAULT_SLIPPAGE_BPS,
    };
    if slippage_bps < 0.0 {
        return Err(KernelError::ExecutionFailed);
    }

    let fx_fee_rate = match config_value(config, "fx_fee_rate") {
        Some(value) => normalize_rate(value)?,
        None => DEFAULT_FX_FEE_RATE,
    };
    let kr_sell_tax_rate = match config_value(config, "kr_sell_tax_rate") {
        Some(value) => normalize_rate(value)?,
        None => DEFAULT_KR_SELL_TAX_RATE,
    };
    let jp_realized_gains_tax_rate = match config_value(config, "jp_realized_gains_tax_rate") {
        Some(value) => normalize_rate(value)?,
        None => DEFAULT_JP_REALIZED_GAINS_TAX_RATE,
    };
    let us_realized_gains_tax_rate = match config_value(config, "us_realized_gains_tax_rate") {
        Some(value) => normalize_rate(value)?,
        None => DEFAULT_US_REALIZED_GAINS_TAX_RATE,
    };

    Ok(CostTaxConfig {
        commission_rate,
        slippage_rate: slippage_bps / 10_000.0,
        fx_fee_rate,
        kr_sell_tax_rate,
        jp_realized_gains_tax_rate,
        us_realized_gains_tax_rate,
    })
}

fn realized_gains_tax_rate(country: &str, config: &CostTaxConfig) -> f64 {
    match country.to_ascii_uppercase().as_str() {
        "JP" => config.jp_realized_gains_tax_rate,
        "US" => config.us_realized_gains_tax_rate,
        _ => 0.0,
    }
}

fn price_date_key(row: &JsonMap, field_name: &str) -> Result<String, KernelError> {
    if let Some(value) = row.get(field_name) {
        return to_iso_date(value);
    }
    let fallback = row.get("as_of_date").ok_or(KernelError::ExecutionFailed)?;
    to_iso_date(fallback)
}

fn build_price_lookup(
    price_rows: &[JsonMap],
) -> Result<(HashMap<(String, String), JsonMap>, HashMap<String, (String, String)>), KernelError> {
    let mut lookup: HashMap<(String, String), JsonMap> = HashMap::new();
    let mut metadata: HashMap<String, (String, String)> = HashMap::new();

    for row in price_rows {
        let security_id = row
            .get("security_id")
            .map(value_to_text)
            .unwrap_or_default();
        if security_id.is_empty() {
            return Err(KernelError::ExecutionFailed);
        }
        let key = (security_id.clone(), price_date_key(row, "price_date")?);
        lookup.insert(key, row.clone());

        let country = row
            .get("country")
            .map(value_to_upper_text)
            .unwrap_or_default();
        let currency = row
            .get("currency")
            .map(value_to_upper_text)
            .unwrap_or_default();
        if !country.is_empty() && !currency.is_empty() {
            metadata.entry(security_id).or_insert((country, currency));
        }
    }

    Ok((lookup, metadata))
}

fn build_benchmark_lookup(benchmark_rows: &[JsonMap]) -> Result<HashMap<String, JsonMap>, KernelError> {
    let mut lookup: HashMap<String, JsonMap> = HashMap::new();
    for row in benchmark_rows {
        if !row.contains_key("close") {
            return Err(KernelError::ExecutionFailed);
        }
        let benchmark_date = price_date_key(row, "benchmark_date")?;
        lookup.insert(benchmark_date, row.clone());
    }
    Ok(lookup)
}

fn normalize_allocations(value: &Value) -> Result<HashMap<String, Vec<Allocation>>, KernelError> {
    let mut grouped: BTreeMap<String, Vec<Allocation>> = BTreeMap::new();

    match value {
        Value::Object(mapping_rows) => {
            for (raw_date, rows_value) in mapping_rows {
                let rebalance_date = to_iso_date(&Value::String(raw_date.clone()))?;
                let rows = rows_value.as_array().ok_or(KernelError::ExecutionFailed)?;
                let bucket = grouped.entry(rebalance_date).or_default();

                for row_value in rows {
                    let row = row_value.as_object().ok_or(KernelError::ExecutionFailed)?;
                    let security_id = row
                        .get("security_id")
                        .map(value_to_text)
                        .unwrap_or_default();
                    if security_id.is_empty() {
                        return Err(KernelError::ExecutionFailed);
                    }
                    let weight_value = row
                        .get("target_weight")
                        .or_else(|| row.get("weight"))
                        .ok_or(KernelError::ExecutionFailed)?;
                    let weight = to_float(weight_value)?;
                    if weight < 0.0 {
                        return Err(KernelError::ExecutionFailed);
                    }
                    bucket.push(Allocation {
                        security_id,
                        target_weight: weight,
                    });
                }
            }
        }
        Value::Array(rows) => {
            for row_value in rows {
                let row = row_value.as_object().ok_or(KernelError::ExecutionFailed)?;
                let rebalance_date_value = row
                    .get("rebalance_date")
                    .ok_or(KernelError::ExecutionFailed)?;
                let rebalance_date = to_iso_date(rebalance_date_value)?;
                let security_id = row
                    .get("security_id")
                    .map(value_to_text)
                    .unwrap_or_default();
                if security_id.is_empty() {
                    return Err(KernelError::ExecutionFailed);
                }
                let weight_value = row
                    .get("target_weight")
                    .or_else(|| row.get("weight"))
                    .ok_or(KernelError::ExecutionFailed)?;
                let weight = to_float(weight_value)?;
                if weight < 0.0 {
                    return Err(KernelError::ExecutionFailed);
                }

                grouped
                    .entry(rebalance_date)
                    .or_default()
                    .push(Allocation {
                        security_id,
                        target_weight: weight,
                    });
            }
        }
        _ => return Err(KernelError::ExecutionFailed),
    }

    let mut normalized: HashMap<String, Vec<Allocation>> = HashMap::new();
    for (rebalance_date, rows) in grouped {
        let total_weight: f64 = rows.iter().map(|row| row.target_weight).sum();
        if total_weight > 1.0 + FLOAT_EPSILON {
            return Err(KernelError::ExecutionFailed);
        }

        let mut merged: BTreeMap<String, f64> = BTreeMap::new();
        for row in rows {
            *merged.entry(row.security_id).or_insert(0.0) += row.target_weight;
        }
        normalized.insert(
            rebalance_date,
            merged
                .into_iter()
                .map(|(security_id, target_weight)| Allocation {
                    security_id,
                    target_weight,
                })
                .collect(),
        );
    }

    Ok(normalized)
}

fn infer_currency_country(
    security_id: &str,
    positions: &HashMap<String, Position>,
    price_lookup: &HashMap<(String, String), JsonMap>,
    metadata: &HashMap<String, (String, String)>,
    as_of_date: &str,
) -> Result<(String, String), KernelError> {
    if let Some(existing) = positions.get(security_id) {
        return Ok((existing.country.clone(), existing.currency.clone()));
    }
    if let Some((country, currency)) = metadata.get(security_id) {
        return Ok((country.clone(), currency.clone()));
    }

    let key = (security_id.to_string(), as_of_date.to_string());
    let row = price_lookup.get(&key).ok_or(KernelError::ExecutionFailed)?;
    let country = row
        .get("country")
        .map(value_to_upper_text)
        .unwrap_or_default();
    let currency = row
        .get("currency")
        .map(value_to_upper_text)
        .unwrap_or_default();
    if country.is_empty() || currency.is_empty() {
        return Err(KernelError::ExecutionFailed);
    }
    Ok((country, currency))
}

fn fx_rate_on_date(rows: &[JsonMap], pair: &str, fx_date: &str) -> Result<f64, KernelError> {
    let normalized_pair = pair.trim().to_ascii_uppercase();
    let lookup_date = to_iso_date(&Value::String(fx_date.to_string()))?;

    let mut date_to_rate: HashMap<String, f64> = HashMap::new();
    for row in rows {
        let row_pair = row
            .get("pair")
            .map(value_to_upper_text)
            .unwrap_or_default();
        if row_pair != normalized_pair {
            continue;
        }
        let row_date_value = row.get("fx_date").ok_or(KernelError::ExecutionFailed)?;
        let row_date = to_iso_date(row_date_value)?;
        let rate_value = row.get("rate").ok_or(KernelError::ExecutionFailed)?;
        let rate = to_float(rate_value)?;
        date_to_rate.insert(row_date, rate);
    }

    if date_to_rate.is_empty() {
        return Err(KernelError::ExecutionFailed);
    }

    if let Some(rate) = date_to_rate.get(&lookup_date) {
        return Ok(*rate);
    }

    let mut prior_dates: Vec<&String> = date_to_rate
        .keys()
        .filter(|row_date| *row_date <= &lookup_date)
        .collect();
    if prior_dates.is_empty() {
        return Err(KernelError::ExecutionFailed);
    }
    prior_dates.sort();
    let nearest = prior_dates.last().ok_or(KernelError::ExecutionFailed)?;
    date_to_rate
        .get(*nearest)
        .copied()
        .ok_or(KernelError::ExecutionFailed)
}

fn to_krw_price(
    price_local: f64,
    currency: &str,
    as_of_date: &str,
    fx_rates: &[JsonMap],
) -> Result<f64, KernelError> {
    let normalized = currency.to_ascii_uppercase();
    if normalized == "KRW" {
        return Ok(price_local);
    }

    let usd_krw = fx_rate_on_date(fx_rates, "USD/KRW", as_of_date)?;
    if normalized == "USD" {
        return Ok(price_local * usd_krw);
    }
    if normalized == "JPY" {
        let usd_jpy = fx_rate_on_date(fx_rates, "USD/JPY", as_of_date)?;
        return Ok(price_local * (usd_krw / usd_jpy));
    }
    Err(KernelError::ExecutionFailed)
}

fn price_krw_for_security(
    security_id: &str,
    as_of_date: &str,
    currency: &str,
    price_lookup: &HashMap<(String, String), JsonMap>,
    fx_rates: &[JsonMap],
) -> Result<(f64, f64), KernelError> {
    let key = (security_id.to_string(), as_of_date.to_string());
    let row = price_lookup.get(&key).ok_or(KernelError::ExecutionFailed)?;
    let local_price = to_float(row.get("close").ok_or(KernelError::ExecutionFailed)?)?;
    let krw_price = to_krw_price(local_price, currency, as_of_date, fx_rates)?;
    Ok((local_price, krw_price))
}

fn fx_return_to_krw(
    currency: &str,
    start_usd_krw: f64,
    end_usd_krw: f64,
    start_usd_jpy: Option<f64>,
    end_usd_jpy: Option<f64>,
) -> Result<f64, KernelError> {
    let normalized = currency.to_ascii_uppercase();
    if normalized == "KRW" {
        return Ok(0.0);
    }
    if normalized == "USD" {
        return Ok((end_usd_krw / start_usd_krw) - 1.0);
    }
    if normalized == "JPY" {
        let start = start_usd_jpy.ok_or(KernelError::ExecutionFailed)?;
        let end = end_usd_jpy.ok_or(KernelError::ExecutionFailed)?;
        let start_jpy_krw = start_usd_krw / start;
        let end_jpy_krw = end_usd_krw / end;
        return Ok((end_jpy_krw / start_jpy_krw) - 1.0);
    }
    Err(KernelError::ExecutionFailed)
}

fn convert_return_to_krw_base(
    local_return: f64,
    currency: &str,
    start_usd_krw: f64,
    end_usd_krw: f64,
    start_usd_jpy: Option<f64>,
    end_usd_jpy: Option<f64>,
) -> Result<f64, KernelError> {
    let fx_return = fx_return_to_krw(
        currency,
        start_usd_krw,
        end_usd_krw,
        start_usd_jpy,
        end_usd_jpy,
    )?;
    Ok(((1.0 + local_return) * (1.0 + fx_return)) - 1.0)
}

fn benchmark_return_krw(
    start_date: &str,
    end_date: &str,
    benchmark_lookup: &HashMap<String, JsonMap>,
    fx_rates: &[JsonMap],
) -> Result<Option<f64>, KernelError> {
    let Some(start) = benchmark_lookup.get(start_date) else {
        return Ok(None);
    };
    let Some(end) = benchmark_lookup.get(end_date) else {
        return Ok(None);
    };

    let start_close = to_float(start.get("close").ok_or(KernelError::ExecutionFailed)?)?;
    let end_close = to_float(end.get("close").ok_or(KernelError::ExecutionFailed)?)?;
    if start_close == 0.0 {
        return Err(KernelError::ExecutionFailed);
    }

    let local_return = (end_close / start_close) - 1.0;
    let currency = start
        .get("currency")
        .map(value_to_upper_text)
        .unwrap_or_default();
    if currency.is_empty() {
        return Err(KernelError::ExecutionFailed);
    }

    let start_usd_krw = fx_rate_on_date(fx_rates, "USD/KRW", start_date)?;
    let end_usd_krw = fx_rate_on_date(fx_rates, "USD/KRW", end_date)?;

    let mut start_usd_jpy: Option<f64> = None;
    let mut end_usd_jpy: Option<f64> = None;
    if currency == "JPY" {
        start_usd_jpy = Some(fx_rate_on_date(fx_rates, "USD/JPY", start_date)?);
        end_usd_jpy = Some(fx_rate_on_date(fx_rates, "USD/JPY", end_date)?);
    }

    Ok(Some(convert_return_to_krw_base(
        local_return,
        currency.as_str(),
        start_usd_krw,
        end_usd_krw,
        start_usd_jpy,
        end_usd_jpy,
    )?))
}

fn sorted_keys<T>(map: &HashMap<String, T>) -> Vec<String> {
    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();
    keys
}

fn run_backtest_core(request: BacktestKernelRequest) -> Result<BacktestResult, KernelError> {
    let mut schedule: Vec<String> = request
        .rebalance_schedule
        .iter()
        .map(to_iso_date)
        .collect::<Result<Vec<_>, _>>()?;
    schedule.sort();
    schedule.dedup();

    if schedule.len() < 2 {
        return Err(KernelError::ExecutionFailed);
    }
    if request.initial_nav_krw <= 0.0 {
        return Err(KernelError::ExecutionFailed);
    }

    let config = parse_cost_tax_config(request.cost_tax_config.as_ref())?;
    let (price_lookup, metadata) = build_price_lookup(&request.prices)?;
    let benchmark_lookup = build_benchmark_lookup(&request.benchmark_series)?;
    let allocation_lookup = normalize_allocations(&request.portfolio_allocations)?;

    let mut trades: Vec<TradeLedgerEntry> = Vec::new();
    let mut holdings_history: Vec<HoldingSnapshot> = Vec::new();
    let mut period_returns: Vec<PeriodReturn> = Vec::new();

    let mut positions: HashMap<String, Position> = HashMap::new();
    let mut cash_krw = request.initial_nav_krw;
    let mut benchmark_growth = 1.0;

    for period_index in 0..(schedule.len() - 1) {
        let start_date = &schedule[period_index];
        let end_date = &schedule[period_index + 1];

        let allocations = allocation_lookup
            .get(start_date)
            .cloned()
            .unwrap_or_default();

        let mut current_values: HashMap<String, f64> = HashMap::new();
        for security_id in sorted_keys(&positions) {
            let position = positions
                .get(security_id.as_str())
                .cloned()
                .ok_or(KernelError::ExecutionFailed)?;
            let (_, krw_price) = price_krw_for_security(
                security_id.as_str(),
                start_date,
                position.currency.as_str(),
                &price_lookup,
                &request.fx_rates,
            )?;
            current_values.insert(security_id, position.quantity * krw_price);
        }

        let start_nav_krw = cash_krw + current_values.values().sum::<f64>();

        let target_weights: HashMap<String, f64> = allocations
            .iter()
            .map(|row| (row.security_id.clone(), row.target_weight))
            .collect();

        let mut universe: Vec<String> = positions
            .keys()
            .cloned()
            .chain(target_weights.keys().cloned())
            .collect();
        universe.sort();
        universe.dedup();

        let mut sell_candidates: Vec<(String, f64)> = Vec::new();
        let mut buy_candidates: Vec<(String, f64)> = Vec::new();

        for security_id in universe {
            let target_value = start_nav_krw * target_weights.get(security_id.as_str()).copied().unwrap_or(0.0);
            let current_value = current_values.get(security_id.as_str()).copied().unwrap_or(0.0);
            let delta = target_value - current_value;
            if delta.abs() <= TARGET_DELTA_EPSILON {
                continue;
            }
            if delta < 0.0 {
                sell_candidates.push((security_id, delta.abs()));
            } else {
                buy_candidates.push((security_id, delta));
            }
        }

        let mut period_commission = 0.0;
        let mut period_slippage = 0.0;
        let mut period_fx_fee = 0.0;
        let mut period_sell_tax = 0.0;
        let mut period_realized_gains_tax = 0.0;

        for (security_id, sell_value) in sell_candidates {
            let Some(position) = positions.get(security_id.as_str()).cloned() else {
                continue;
            };

            let (local_price, krw_price) = price_krw_for_security(
                security_id.as_str(),
                start_date,
                position.currency.as_str(),
                &price_lookup,
                &request.fx_rates,
            )?;

            let max_notional = position.quantity * krw_price;
            let notional_krw = sell_value.min(max_notional);
            let quantity = if krw_price > 0.0 { notional_krw / krw_price } else { 0.0 };
            if quantity <= 0.0 {
                continue;
            }

            let realized_gain = ((krw_price - position.avg_cost_krw) * quantity).max(0.0);
            let realized_gains_tax =
                realized_gain * realized_gains_tax_rate(position.country.as_str(), &config);
            let sell_tax = if position.country == "KR" {
                notional_krw * config.kr_sell_tax_rate
            } else {
                0.0
            };
            let commission = notional_krw * config.commission_rate;
            let slippage = notional_krw * config.slippage_rate;
            let fx_fee = if position.currency != "KRW" {
                notional_krw * config.fx_fee_rate
            } else {
                0.0
            };
            let total_cost_tax = commission + slippage + fx_fee + sell_tax + realized_gains_tax;

            cash_krw += notional_krw - total_cost_tax;

            let mut remove_position = false;
            if let Some(entry) = positions.get_mut(security_id.as_str()) {
                entry.quantity -= quantity;
                remove_position = entry.quantity <= FLOAT_EPSILON;
            }
            if remove_position {
                positions.remove(security_id.as_str());
            }

            period_commission += commission;
            period_slippage += slippage;
            period_fx_fee += fx_fee;
            period_sell_tax += sell_tax;
            period_realized_gains_tax += realized_gains_tax;

            trades.push(TradeLedgerEntry {
                rebalance_date: start_date.clone(),
                security_id,
                side: "SELL".to_string(),
                country: position.country,
                currency: position.currency,
                quantity,
                price_local: local_price,
                price_krw: krw_price,
                notional_krw,
                gross_proceeds_krw: notional_krw,
                gross_cost_krw: 0.0,
                commission_krw: commission,
                slippage_krw: slippage,
                fx_fee_krw: fx_fee,
                sell_tax_krw: sell_tax,
                realized_gain_krw: realized_gain,
                realized_gains_tax_krw: realized_gains_tax,
                total_cost_tax_krw: total_cost_tax,
            });
        }

        for (security_id, buy_value) in buy_candidates {
            let (country, currency) = infer_currency_country(
                security_id.as_str(),
                &positions,
                &price_lookup,
                &metadata,
                start_date,
            )?;

            let (local_price, krw_price) = price_krw_for_security(
                security_id.as_str(),
                start_date,
                currency.as_str(),
                &price_lookup,
                &request.fx_rates,
            )?;
            if krw_price <= 0.0 {
                return Err(KernelError::ExecutionFailed);
            }

            let notional_krw = buy_value;
            let commission = notional_krw * config.commission_rate;
            let slippage = notional_krw * config.slippage_rate;
            let fx_fee = if currency != "KRW" {
                notional_krw * config.fx_fee_rate
            } else {
                0.0
            };
            let total_cost_tax = commission + slippage + fx_fee;
            let quantity = notional_krw / krw_price;

            cash_krw -= notional_krw + total_cost_tax;

            if let Some(existing) = positions.get(security_id.as_str()).cloned() {
                let new_quantity = existing.quantity + quantity;
                if new_quantity <= 0.0 {
                    return Err(KernelError::ExecutionFailed);
                }
                let weighted_avg_cost =
                    ((existing.quantity * existing.avg_cost_krw) + notional_krw) / new_quantity;
                positions.insert(
                    security_id.clone(),
                    Position {
                        country: existing.country,
                        currency: existing.currency,
                        quantity: new_quantity,
                        avg_cost_krw: weighted_avg_cost,
                    },
                );
            } else {
                positions.insert(
                    security_id.clone(),
                    Position {
                        country: country.clone(),
                        currency: currency.clone(),
                        quantity,
                        avg_cost_krw: krw_price,
                    },
                );
            }

            period_commission += commission;
            period_slippage += slippage;
            period_fx_fee += fx_fee;

            trades.push(TradeLedgerEntry {
                rebalance_date: start_date.clone(),
                security_id,
                side: "BUY".to_string(),
                country,
                currency,
                quantity,
                price_local: local_price,
                price_krw: krw_price,
                notional_krw,
                gross_proceeds_krw: 0.0,
                gross_cost_krw: notional_krw,
                commission_krw: commission,
                slippage_krw: slippage,
                fx_fee_krw: fx_fee,
                sell_tax_krw: 0.0,
                realized_gain_krw: 0.0,
                realized_gains_tax_krw: 0.0,
                total_cost_tax_krw: total_cost_tax,
            });
        }

        let mut end_nav_krw = cash_krw;
        let mut end_values: Vec<(String, Position, f64, f64, f64)> = Vec::new();
        for security_id in sorted_keys(&positions) {
            let position = positions
                .get(security_id.as_str())
                .cloned()
                .ok_or(KernelError::ExecutionFailed)?;
            let (local_price, krw_price) = price_krw_for_security(
                security_id.as_str(),
                end_date,
                position.currency.as_str(),
                &price_lookup,
                &request.fx_rates,
            )?;
            let market_value = position.quantity * krw_price;
            end_nav_krw += market_value;
            end_values.push((security_id, position, local_price, krw_price, market_value));
        }

        for (security_id, position, local_price, krw_price, market_value) in end_values {
            let weight = if end_nav_krw == 0.0 {
                0.0
            } else {
                market_value / end_nav_krw
            };
            holdings_history.push(HoldingSnapshot {
                as_of_date: end_date.clone(),
                security_id,
                country: position.country,
                currency: position.currency,
                quantity: position.quantity,
                avg_cost_krw: position.avg_cost_krw,
                price_local: local_price,
                price_krw: krw_price,
                market_value_krw: market_value,
                weight,
            });
        }

        let total_cost_tax = period_commission
            + period_slippage
            + period_fx_fee
            + period_sell_tax
            + period_realized_gains_tax;
        let gross_end_nav = end_nav_krw + total_cost_tax;
        let net_return = (end_nav_krw / start_nav_krw) - 1.0;
        let gross_return = (gross_end_nav / start_nav_krw) - 1.0;

        let benchmark_return =
            benchmark_return_krw(start_date, end_date, &benchmark_lookup, &request.fx_rates)?;
        if let Some(value) = benchmark_return {
            benchmark_growth *= 1.0 + value;
        }

        period_returns.push(PeriodReturn {
            start_date: start_date.clone(),
            end_date: end_date.clone(),
            start_nav_krw,
            end_nav_krw,
            gross_return,
            net_return,
            benchmark_krw_return: benchmark_return,
            total_commission_krw: period_commission,
            total_slippage_krw: period_slippage,
            total_fx_fee_krw: period_fx_fee,
            total_sell_tax_krw: period_sell_tax,
            total_realized_gains_tax_krw: period_realized_gains_tax,
            total_cost_tax_krw: total_cost_tax,
        });
    }

    let total_commission: f64 = period_returns.iter().map(|row| row.total_commission_krw).sum();
    let total_slippage: f64 = period_returns.iter().map(|row| row.total_slippage_krw).sum();
    let total_fx_fee: f64 = period_returns.iter().map(|row| row.total_fx_fee_krw).sum();
    let total_sell_tax: f64 = period_returns.iter().map(|row| row.total_sell_tax_krw).sum();
    let total_realized_gains_tax: f64 = period_returns
        .iter()
        .map(|row| row.total_realized_gains_tax_krw)
        .sum();
    let total_cost_tax: f64 = period_returns.iter().map(|row| row.total_cost_tax_krw).sum();

    let end_nav = period_returns
        .last()
        .map(|row| row.end_nav_krw)
        .unwrap_or(request.initial_nav_krw);
    let cumulative_net_return = (end_nav / request.initial_nav_krw) - 1.0;
    let cumulative_gross_return = ((end_nav + total_cost_tax) / request.initial_nav_krw) - 1.0;
    let benchmark_cumulative_return = if period_returns
        .iter()
        .all(|row| row.benchmark_krw_return.is_some())
    {
        Some(benchmark_growth - 1.0)
    } else {
        None
    };

    let metrics = BacktestMetrics {
        periods: period_returns.len(),
        start_nav_krw: request.initial_nav_krw,
        end_nav_krw: end_nav,
        cumulative_net_return,
        cumulative_gross_return,
        benchmark_cumulative_return,
        total_commission_krw: total_commission,
        total_slippage_krw: total_slippage,
        total_fx_fee_krw: total_fx_fee,
        total_sell_tax_krw: total_sell_tax,
        total_realized_gains_tax_krw: total_realized_gains_tax,
        total_cost_tax_krw: total_cost_tax,
    };

    Ok(BacktestResult {
        trades,
        holdings: holdings_history,
        returns: period_returns,
        metrics,
    })
}

fn ranking_compute_raw_metric_for_model(
    model: &str,
    rd_expense: f64,
    sales_ttm: Option<f64>,
    liquidity_krw: f64,
) -> Result<f64, KernelError> {
    let safe_sales = sales_ttm.map(|value| value.max(1e-9));
    let safe_liquidity = liquidity_krw.max(1.0);
    let liquidity_scale = (safe_liquidity / 1_000_000_000.0).ln_1p().max(1e-6);
    let sales_ratio = safe_sales.map(|value| rd_expense / value);
    let sales_size_ratio = sales_ratio.map(|value| value / liquidity_scale);
    let mktcap_proxy_ratio = rd_expense / safe_liquidity;
    let ev_like_ratio = rd_expense / (safe_liquidity * (1.0 + liquidity_scale));

    match model {
        "rnd_sales_ttm" => sales_ratio.ok_or(KernelError::ExecutionFailed),
        "rnd_sales_size_proxy" => sales_size_ratio.ok_or(KernelError::ExecutionFailed),
        "rnd_mktcap_proxy" => Ok(mktcap_proxy_ratio),
        "rnd_ev_proxy" => Ok(ev_like_ratio),
        "rnd_robust_composite" => {
            let mut components = vec![mktcap_proxy_ratio, ev_like_ratio];
            if let Some(value) = sales_ratio {
                components.push(value);
            }
            if let Some(value) = sales_size_ratio {
                components.push(value);
            }
            Ok(components.iter().sum::<f64>() / components.len() as f64)
        }
        _ => Err(KernelError::ExecutionFailed),
    }
}

fn ranking_score_weights(model: &str) -> (f64, f64) {
    match model {
        "rnd_sales_ttm" => (0.7, 0.3),
        "rnd_sales_size_proxy" => (0.4, 0.6),
        "rnd_mktcap_proxy" => (0.3, 0.7),
        "rnd_ev_proxy" => (0.6, 0.4),
        "rnd_robust_composite" => (0.5, 0.5),
        _ => (0.5, 0.5),
    }
}

fn ranking_zscore_by_key(values: &[(String, f64)]) -> HashMap<String, f64> {
    if values.is_empty() {
        return HashMap::new();
    }

    let mean_value = values.iter().map(|(_, value)| *value).sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|(_, value)| {
            let diff = *value - mean_value;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    let std_value = variance.sqrt();
    if std_value <= 1e-12 {
        return values
            .iter()
            .map(|(key, _)| (key.clone(), 0.0))
            .collect();
    }

    values
        .iter()
        .map(|(key, value)| (key.clone(), (*value - mean_value) / std_value))
        .collect()
}

fn ranking_sort_indices_by_factor(rows: &[RankingCandidate], row_indices: &mut Vec<usize>) {
    row_indices.sort_by(|left, right| {
        rows[*right]
            .factor_value
            .total_cmp(&rows[*left].factor_value)
            .then_with(|| rows[*left].security_id.cmp(&rows[*right].security_id))
    });
}

fn ranking_compute_sector_quotas(
    total: usize,
    sector_counts: &HashMap<String, usize>,
) -> HashMap<String, usize> {
    if total == 0 || sector_counts.is_empty() {
        return HashMap::new();
    }

    let total_count: usize = sector_counts.values().sum();
    if total_count == 0 {
        return sector_counts
            .keys()
            .map(|sector| (sector.clone(), 0usize))
            .collect();
    }

    let raw: HashMap<String, f64> = sector_counts
        .iter()
        .map(|(sector, count)| {
            (
                sector.clone(),
                total as f64 * (*count as f64 / total_count as f64),
            )
        })
        .collect();
    let mut quotas: HashMap<String, usize> = sector_counts
        .keys()
        .map(|sector| {
            (
                sector.clone(),
                raw.get(sector).copied().unwrap_or(0.0).floor() as usize,
            )
        })
        .collect();

    if total >= sector_counts.len() {
        for sector in sector_counts.keys() {
            if quotas.get(sector).copied().unwrap_or(0) == 0 {
                quotas.insert(sector.clone(), 1);
            }
        }
    }

    let used: usize = quotas.values().sum();
    if used > total {
        let mut overflow = used - total;
        let mut sectors: Vec<String> = quotas.keys().cloned().collect();
        sectors.sort_by(|left, right| {
            quotas
                .get(right)
                .copied()
                .unwrap_or(0)
                .cmp(&quotas.get(left).copied().unwrap_or(0))
                .then_with(|| left.cmp(right))
        });
        for sector in sectors {
            if overflow == 0 {
                break;
            }
            let current = quotas.get(&sector).copied().unwrap_or(0);
            let reducible = current.saturating_sub(1);
            if reducible == 0 {
                continue;
            }
            let step = reducible.min(overflow);
            quotas.insert(sector.clone(), current - step);
            overflow -= step;
        }
    } else if used < total {
        let mut shortfall = total - used;
        let mut sectors: Vec<String> = sector_counts.keys().cloned().collect();
        sectors.sort_by(|left, right| {
            let left_remainder = raw.get(left).copied().unwrap_or(0.0)
                - quotas.get(left).copied().unwrap_or(0) as f64;
            let right_remainder = raw.get(right).copied().unwrap_or(0.0)
                - quotas.get(right).copied().unwrap_or(0) as f64;
            right_remainder
                .total_cmp(&left_remainder)
                .then_with(|| left.cmp(right))
        });
        for sector in sectors {
            if shortfall == 0 {
                break;
            }
            let next = quotas.get(&sector).copied().unwrap_or(0) + 1;
            quotas.insert(sector, next);
            shortfall -= 1;
        }
    }

    quotas
}

fn ranking_benchmark_sector_weights(
    rows: &[RankingCandidate],
    row_indices: &[usize],
) -> HashMap<String, f64> {
    let mut sector_liquidity: HashMap<String, f64> = HashMap::new();
    for row_index in row_indices {
        let row = &rows[*row_index];
        let liquidity = row.median_traded_value_krw.max(1.0);
        *sector_liquidity.entry(row.sector.clone()).or_insert(0.0) += liquidity;
    }

    let total = sector_liquidity.values().sum::<f64>();
    if total <= 0.0 {
        return HashMap::new();
    }

    sector_liquidity
        .into_iter()
        .map(|(sector, liquidity)| (sector, liquidity / total))
        .collect()
}

fn ranking_compute_sector_quota_bounds(
    total: usize,
    sector_counts: &HashMap<String, usize>,
    sector_weights: &HashMap<String, f64>,
    active_band: f64,
) -> (HashMap<String, usize>, HashMap<String, usize>) {
    if total == 0 || sector_counts.is_empty() {
        return (HashMap::new(), HashMap::new());
    }

    let normalized_band = active_band.clamp(0.0, 1.0);
    let mut min_quota: HashMap<String, usize> = HashMap::new();
    let mut max_quota: HashMap<String, usize> = HashMap::new();

    for (sector, available) in sector_counts {
        let weight = sector_weights.get(sector).copied().unwrap_or(0.0);
        let lower_weight = (weight - normalized_band).max(0.0);
        let upper_weight = (weight + normalized_band).min(1.0);
        let mut minimum = (total as f64 * lower_weight + 1e-12).floor() as usize;
        let mut maximum = (total as f64 * upper_weight - 1e-12).ceil() as usize;
        minimum = minimum.min(*available);
        if maximum < minimum {
            maximum = minimum;
        }
        maximum = maximum.min(*available);
        min_quota.insert(sector.clone(), minimum);
        max_quota.insert(sector.clone(), maximum);
    }

    let minimum_total: usize = min_quota.values().sum();
    if minimum_total > total {
        let mut overflow = minimum_total - total;
        let mut sectors: Vec<String> = min_quota.keys().cloned().collect();
        sectors.sort_by(|left, right| {
            min_quota
                .get(left)
                .copied()
                .unwrap_or(0)
                .cmp(&min_quota.get(right).copied().unwrap_or(0))
                .then_with(|| left.cmp(right))
        });
        sectors.reverse();
        for sector in sectors {
            if overflow == 0 {
                break;
            }
            let current = min_quota.get(&sector).copied().unwrap_or(0);
            if current == 0 {
                continue;
            }
            let step = current.min(overflow);
            min_quota.insert(sector, current - step);
            overflow -= step;
        }
    }

    (min_quota, max_quota)
}

fn ranking_select_sector_rows(
    rows: &[RankingCandidate],
    row_indices: &[usize],
    pick_count: usize,
    use_size_stratification: bool,
) -> Vec<usize> {
    if pick_count == 0 || row_indices.is_empty() {
        return Vec::new();
    }

    let mut ordered_rows = row_indices.to_vec();
    ranking_sort_indices_by_factor(rows, &mut ordered_rows);
    if !use_size_stratification || pick_count >= ordered_rows.len() {
        return ordered_rows.into_iter().take(pick_count).collect();
    }

    let mut size_groups: HashMap<String, Vec<usize>> = HashMap::new();
    let mut bucket_order: Vec<String> = Vec::new();
    for row_index in &ordered_rows {
        let bucket = rows[*row_index].size_bucket.clone();
        if !size_groups.contains_key(&bucket) {
            bucket_order.push(bucket.clone());
        }
        size_groups.entry(bucket).or_default().push(*row_index);
    }
    for bucket_rows in size_groups.values_mut() {
        ranking_sort_indices_by_factor(rows, bucket_rows);
    }

    let bucket_counts: HashMap<String, usize> = size_groups
        .iter()
        .map(|(bucket, bucket_rows)| (bucket.clone(), bucket_rows.len()))
        .collect();
    let bucket_quotas = ranking_compute_sector_quotas(pick_count, &bucket_counts);

    let mut selected: Vec<usize> = Vec::new();
    for bucket in bucket_order {
        let quota = bucket_quotas.get(&bucket).copied().unwrap_or(0);
        if quota == 0 {
            continue;
        }
        if let Some(bucket_rows) = size_groups.get(&bucket) {
            selected.extend(bucket_rows.iter().copied().take(quota));
        }
    }

    if selected.len() < pick_count {
        let selected_ids: HashSet<String> = selected
            .iter()
            .map(|row_index| rows[*row_index].security_id.clone())
            .collect();
        let leftovers: Vec<usize> = ordered_rows
            .iter()
            .copied()
            .filter(|row_index| !selected_ids.contains(&rows[*row_index].security_id))
            .collect();
        selected.extend(leftovers.into_iter().take(pick_count - selected.len()));
    }

    selected.truncate(pick_count);
    selected
}

fn parse_ranking_accepted_liquidity(rows: &[JsonMap]) -> Result<HashMap<String, f64>, KernelError> {
    let mut accepted_by_security: HashMap<String, f64> = HashMap::new();
    for row in rows {
        let security_id = row
            .get("security_id")
            .map(value_to_text)
            .unwrap_or_default();
        if security_id.is_empty() {
            return Err(KernelError::ExecutionFailed);
        }
        let liquidity = to_float(
            row.get("median_traded_value_krw")
                .ok_or(KernelError::ExecutionFailed)?,
        )?;
        accepted_by_security.insert(security_id, liquidity);
    }
    Ok(accepted_by_security)
}

fn parse_ranking_factor_rows(
    request: &RankingKernelRequest,
    accepted_by_security: &HashMap<String, f64>,
) -> Result<Vec<RankingCandidate>, KernelError> {
    let mut eligible_rows: Vec<RankingCandidate> = Vec::new();
    for row in &request.factor_rows {
        let security_id = row
            .get("security_id")
            .map(value_to_text)
            .unwrap_or_default();
        if security_id.is_empty() {
            return Err(KernelError::ExecutionFailed);
        }

        let factor_value = to_optional_float_value(row.get("factor_value"))?;
        let rd_expense = to_optional_float_value(row.get("rd_expense"))?;
        let sales_ttm = to_optional_float_value(row.get("sales_ttm"))?;
        let is_eligible = to_optional_bool_value(row.get("is_eligible"))?.unwrap_or(false);
        if !is_eligible || factor_value.is_none() {
            continue;
        }

        let Some(rd_expense_value) = rd_expense else {
            continue;
        };
        let Some(liquidity) = accepted_by_security.get(&security_id).copied() else {
            continue;
        };

        let raw_metric = match ranking_compute_raw_metric_for_model(
            request.factor_model.as_str(),
            rd_expense_value,
            sales_ttm,
            liquidity,
        ) {
            Ok(value) => value,
            Err(_) => continue,
        };

        let country = row.get("country").map(value_to_text).unwrap_or_default();
        let sector = request
            .sector_by_security
            .get(&security_id)
            .cloned()
            .unwrap_or_else(|| "UNKNOWN".to_string());
        eligible_rows.push(RankingCandidate {
            security_id,
            country,
            factor_value: raw_metric,
            base_factor_value: raw_metric,
            sector,
            median_traded_value_krw: liquidity,
            rd_expense: rd_expense_value,
            sales_ttm,
            size_bucket: "mid".to_string(),
            rank_in_country: None,
            is_eligible: false,
        });
    }
    Ok(eligible_rows)
}

fn run_ranking_core(request: RankingKernelRequest) -> Result<Vec<RankingCandidate>, KernelError> {
    if !request.sector_active_band.is_finite() {
        return Err(KernelError::ExecutionFailed);
    }

    let accepted_by_security = parse_ranking_accepted_liquidity(&request.accepted_rows)?;
    let mut eligible_rows = parse_ranking_factor_rows(&request, &accepted_by_security)?;

    let mut groups_by_country_sector: HashMap<(String, String), Vec<usize>> = HashMap::new();
    let mut groups_by_country: HashMap<String, Vec<usize>> = HashMap::new();
    for (row_index, row) in eligible_rows.iter().enumerate() {
        groups_by_country
            .entry(row.country.clone())
            .or_default()
            .push(row_index);
        groups_by_country_sector
            .entry((row.country.clone(), row.sector.clone()))
            .or_default()
            .push(row_index);
    }

    let mut sector_zscores: HashMap<String, f64> = HashMap::new();
    for row_indices in groups_by_country_sector.values() {
        let values: Vec<(String, f64)> = row_indices
            .iter()
            .map(|row_index| {
                (
                    eligible_rows[*row_index].security_id.clone(),
                    eligible_rows[*row_index].factor_value,
                )
            })
            .collect();
        for (security_id, zscore) in ranking_zscore_by_key(&values) {
            sector_zscores.insert(security_id, zscore);
        }
    }

    let mut size_zscores: HashMap<String, f64> = HashMap::new();
    for (country, row_indices) in &groups_by_country {
        let mut sorted_liquidity: Vec<f64> = row_indices
            .iter()
            .map(|row_index| eligible_rows[*row_index].median_traded_value_krw.max(1.0))
            .collect();
        sorted_liquidity.sort_by(|left, right| left.total_cmp(right));

        let (low_cut, high_cut) = if sorted_liquidity.is_empty() {
            (0.0, 0.0)
        } else {
            let low_index = ((row_indices.len() as isize / 3) - 1).max(0) as usize;
            let high_index = (((2 * row_indices.len()) as isize / 3) - 1).max(0) as usize;
            (
                sorted_liquidity[low_index],
                sorted_liquidity[high_index],
            )
        };

        let mut size_adjusted_values: Vec<(String, f64)> = Vec::new();
        for row_index in row_indices {
            let liquidity = eligible_rows[*row_index].median_traded_value_krw.max(1.0);
            let size_scale = (liquidity / 1_000_000_000.0).ln_1p();
            let adjusted = eligible_rows[*row_index].factor_value / size_scale.max(1e-6);
            size_adjusted_values.push((eligible_rows[*row_index].security_id.clone(), adjusted));
            eligible_rows[*row_index].size_bucket = if liquidity <= low_cut {
                "small".to_string()
            } else if liquidity <= high_cut {
                "mid".to_string()
            } else {
                "large".to_string()
            };
        }

        for (security_id, zscore) in ranking_zscore_by_key(&size_adjusted_values) {
            size_zscores.insert(format!("{country}:{security_id}"), zscore);
        }
    }

    let (sector_weight, size_weight) = ranking_score_weights(request.factor_model.as_str());
    for row in eligible_rows.iter_mut() {
        let sector_z = sector_zscores.get(&row.security_id).copied().unwrap_or(0.0);
        let size_z = size_zscores
            .get(format!("{}:{}", row.country, row.security_id).as_str())
            .copied()
            .unwrap_or(0.0);
        row.factor_value = (sector_weight * sector_z) + (size_weight * size_z);
    }

    let mut ranked_indices: Vec<usize> = Vec::new();
    for (country, row_indices) in &groups_by_country {
        let mut ordered_rows = row_indices.clone();
        ranking_sort_indices_by_factor(&eligible_rows, &mut ordered_rows);
        let mut prioritized_rows = ordered_rows.clone();

        let target_count = request
            .requested_counts_by_country
            .get(country)
            .copied()
            .unwrap_or(0);
        if target_count > 0 && ordered_rows.len() > target_count as usize {
            let target_count_usize = target_count as usize;
            let mut sector_groups: HashMap<String, Vec<usize>> = HashMap::new();
            for row_index in row_indices {
                let sector = eligible_rows[*row_index].sector.clone();
                sector_groups.entry(sector).or_default().push(*row_index);
            }
            for sector_rows in sector_groups.values_mut() {
                ranking_sort_indices_by_factor(&eligible_rows, sector_rows);
            }

            let sector_counts: HashMap<String, usize> = sector_groups
                .iter()
                .map(|(sector, rows)| (sector.clone(), rows.len()))
                .collect();
            let benchmark_sector_weights =
                ranking_benchmark_sector_weights(&eligible_rows, row_indices);
            let (min_quota, max_quota) = ranking_compute_sector_quota_bounds(
                target_count_usize,
                &sector_counts,
                &benchmark_sector_weights,
                request.sector_active_band,
            );

            let mut selected: Vec<usize> = Vec::new();
            let mut selected_ids: HashSet<String> = HashSet::new();
            let mut selected_by_sector: HashMap<String, usize> = sector_groups
                .keys()
                .map(|sector| (sector.clone(), 0usize))
                .collect();

            let mut sector_order: Vec<String> = sector_groups.keys().cloned().collect();
            sector_order.sort_by(|left, right| {
                let left_weight = benchmark_sector_weights.get(left).copied().unwrap_or(0.0);
                let right_weight = benchmark_sector_weights.get(right).copied().unwrap_or(0.0);
                right_weight
                    .total_cmp(&left_weight)
                    .then_with(|| left.cmp(right))
            });

            for sector in sector_order {
                let quota = min_quota.get(&sector).copied().unwrap_or(0);
                let picks = ranking_select_sector_rows(
                    &eligible_rows,
                    sector_groups
                        .get(&sector)
                        .map_or(&[][..], |rows| rows.as_slice()),
                    quota,
                    request.use_size_stratification,
                );
                for row_index in picks {
                    let security_id = eligible_rows[row_index].security_id.clone();
                    if selected_ids.contains(&security_id) {
                        continue;
                    }
                    selected.push(row_index);
                    selected_ids.insert(security_id);
                    if let Some(count) = selected_by_sector.get_mut(&sector) {
                        *count += 1;
                    }
                }
            }

            #[derive(Debug)]
            struct CapacityCandidate {
                row_index: usize,
                sector_cap: usize,
            }

            let mut capacity_candidates: Vec<CapacityCandidate> = Vec::new();
            let mut sorted_sectors: Vec<String> = sector_groups.keys().cloned().collect();
            sorted_sectors.sort();
            for sector in sorted_sectors {
                let cap = max_quota
                    .get(&sector)
                    .copied()
                    .unwrap_or(*sector_counts.get(&sector).unwrap_or(&0));
                if selected_by_sector.get(&sector).copied().unwrap_or(0) >= cap {
                    continue;
                }
                if let Some(sector_rows) = sector_groups.get(&sector) {
                    for row_index in sector_rows {
                        let security_id = eligible_rows[*row_index].security_id.clone();
                        if selected_ids.contains(&security_id) {
                            continue;
                        }
                        capacity_candidates.push(CapacityCandidate {
                            row_index: *row_index,
                            sector_cap: cap,
                        });
                    }
                }
            }

            capacity_candidates.sort_by(|left, right| {
                eligible_rows[right.row_index]
                    .factor_value
                    .total_cmp(&eligible_rows[left.row_index].factor_value)
                    .then_with(|| {
                        eligible_rows[left.row_index]
                            .security_id
                            .cmp(&eligible_rows[right.row_index].security_id)
                    })
            });
            for candidate in capacity_candidates {
                if selected.len() >= target_count_usize {
                    break;
                }
                let sector = eligible_rows[candidate.row_index].sector.clone();
                if selected_by_sector.get(&sector).copied().unwrap_or(0) >= candidate.sector_cap {
                    continue;
                }
                let security_id = eligible_rows[candidate.row_index].security_id.clone();
                if selected_ids.contains(&security_id) {
                    continue;
                }
                selected.push(candidate.row_index);
                selected_ids.insert(security_id);
                *selected_by_sector.entry(sector).or_insert(0) += 1;
            }

            if selected.len() < target_count_usize {
                let leftovers: Vec<usize> = ordered_rows
                    .iter()
                    .copied()
                    .filter(|row_index| {
                        let security_id = &eligible_rows[*row_index].security_id;
                        !selected_ids.contains(security_id)
                    })
                    .collect();
                selected.extend(
                    leftovers
                        .into_iter()
                        .take(target_count_usize - selected.len()),
                );
            }

            let selected_slice = if selected.len() > target_count_usize {
                &selected[..target_count_usize]
            } else {
                selected.as_slice()
            };
            let selected_security_ids: HashSet<String> = selected_slice
                .iter()
                .map(|row_index| eligible_rows[*row_index].security_id.clone())
                .collect();
            prioritized_rows = ordered_rows
                .iter()
                .copied()
                .filter(|row_index| {
                    selected_security_ids.contains(&eligible_rows[*row_index].security_id)
                })
                .chain(ordered_rows.iter().copied().filter(|row_index| {
                    !selected_security_ids.contains(&eligible_rows[*row_index].security_id)
                }))
                .collect();
        }

        for (rank, row_index) in prioritized_rows.iter().enumerate() {
            let row = eligible_rows
                .get_mut(*row_index)
                .ok_or(KernelError::ExecutionFailed)?;
            row.rank_in_country = Some((rank + 1) as i64);
            row.is_eligible = true;
            ranked_indices.push(*row_index);
        }
    }

    ranked_indices.sort_by(|left, right| {
        eligible_rows[*left]
            .country
            .cmp(&eligible_rows[*right].country)
            .then_with(|| {
                eligible_rows[*left]
                    .rank_in_country
                    .unwrap_or(SENTINEL_RANK)
                    .cmp(&eligible_rows[*right].rank_in_country.unwrap_or(SENTINEL_RANK))
            })
            .then_with(|| {
                eligible_rows[*left]
                    .security_id
                    .cmp(&eligible_rows[*right].security_id)
            })
    });

    Ok(ranked_indices
        .into_iter()
        .map(|row_index| eligible_rows[row_index].clone())
        .collect())
}

fn to_optional_float_value(value: Option<&Value>) -> Result<Option<f64>, KernelError> {
    match value {
        None => Ok(None),
        Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if text.trim().is_empty() => Ok(None),
        Some(raw_value) => Ok(Some(to_float(raw_value)?)),
    }
}

fn to_optional_positive_int_value(value: Option<&Value>) -> Result<Option<i64>, KernelError> {
    match value {
        None => Ok(None),
        Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if text.trim().is_empty() => Ok(None),
        Some(Value::Number(number)) => {
            let parsed = number.as_i64().ok_or(KernelError::ExecutionFailed)?;
            if parsed <= 0 {
                return Err(KernelError::ExecutionFailed);
            }
            Ok(Some(parsed))
        }
        Some(Value::String(text)) => {
            let parsed = text
                .trim()
                .parse::<i64>()
                .map_err(|_| KernelError::ExecutionFailed)?;
            if parsed <= 0 {
                return Err(KernelError::ExecutionFailed);
            }
            Ok(Some(parsed))
        }
        Some(_) => Err(KernelError::ExecutionFailed),
    }
}

fn to_optional_bool_value(value: Option<&Value>) -> Result<Option<bool>, KernelError> {
    match value {
        None => Ok(None),
        Some(Value::Null) => Ok(None),
        Some(Value::String(text)) if text.trim().is_empty() => Ok(None),
        Some(Value::Bool(flag)) => Ok(Some(*flag)),
        Some(_) => Err(KernelError::ExecutionFailed),
    }
}

fn to_optional_non_negative_float_value(value: Option<&Value>) -> Result<Option<f64>, KernelError> {
    let parsed = to_optional_float_value(value)?;
    if let Some(value) = parsed {
        if value < 0.0 {
            return Err(KernelError::ExecutionFailed);
        }
    }
    Ok(parsed)
}

fn parse_constructor_rows(rows: &[JsonMap]) -> Result<Vec<ConstructorRankedRow>, KernelError> {
    let mut parsed: Vec<ConstructorRankedRow> = Vec::new();
    for record in rows {
        let is_eligible = to_optional_bool_value(Some(
            record
                .get("is_eligible")
                .ok_or(KernelError::ExecutionFailed)?,
        ))?;
        if matches!(is_eligible, Some(false)) {
            continue;
        }

        let security_id_value = record
            .get("security_id")
            .ok_or(KernelError::ExecutionFailed)?;
        let security_id = value_to_text(security_id_value);
        if security_id.is_empty() {
            return Err(KernelError::ExecutionFailed);
        }

        let country_value = record.get("country").ok_or(KernelError::ExecutionFailed)?;
        let country = value_to_upper_text(country_value);
        if country.is_empty() {
            return Err(KernelError::ExecutionFailed);
        }

        let factor_value = to_optional_float_value(Some(
            record
                .get("factor_value")
                .ok_or(KernelError::ExecutionFailed)?,
        ))?;
        let rank_in_country = to_optional_positive_int_value(Some(
            record
                .get("rank_in_country")
                .ok_or(KernelError::ExecutionFailed)?,
        ))?;
        let benchmark_proxy_weight = to_optional_non_negative_float_value(record.get("benchmark_proxy_weight"))?;
        let median_traded_value_krw =
            to_optional_non_negative_float_value(record.get("median_traded_value_krw"))?;

        if let Some(value) = factor_value {
            parsed.push(ConstructorRankedRow {
                security_id,
                country,
                factor_value: value,
                rank_in_country,
                benchmark_proxy_weight,
                median_traded_value_krw,
            });
        }
    }
    Ok(parsed)
}

fn constructor_requested_counts(
    max_holdings: usize,
    countries: &[String],
    target_weights: &HashMap<String, f64>,
) -> HashMap<String, usize> {
    let mut raw: HashMap<String, f64> = HashMap::new();
    let mut floored: HashMap<String, usize> = HashMap::new();
    for country in countries {
        let target = target_weights.get(country).copied().unwrap_or(0.0);
        let raw_count = max_holdings as f64 * target;
        raw.insert(country.clone(), raw_count);
        floored.insert(country.clone(), raw_count.floor() as usize);
    }

    let allocated: usize = floored.values().sum();
    let remaining = max_holdings.saturating_sub(allocated);
    let mut remainders: Vec<(String, f64)> = countries
        .iter()
        .map(|country| {
            let raw_value = raw.get(country).copied().unwrap_or(0.0);
            let floored_value = floored.get(country).copied().unwrap_or(0) as f64;
            (country.clone(), raw_value - floored_value)
        })
        .collect();
    remainders.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    for (country, _) in remainders.into_iter().take(remaining) {
        if let Some(value) = floored.get_mut(&country) {
            *value += 1;
        }
    }
    floored
}

fn constructor_safe_std(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.max(0.0).sqrt()
}

fn constructor_rescale_with_caps(
    base_weights: &HashMap<String, f64>,
    cap_weights: &HashMap<String, f64>,
    target_total: f64,
) -> HashMap<String, f64> {
    if target_total <= 0.0 {
        return base_weights
            .keys()
            .map(|security_id| (security_id.clone(), 0.0))
            .collect();
    }

    let weights: HashMap<String, f64> = base_weights
        .iter()
        .map(|(security_id, weight)| (security_id.clone(), (*weight).max(0.0)))
        .collect();
    let caps: HashMap<String, f64> = weights
        .keys()
        .map(|security_id| {
            (
                security_id.clone(),
                cap_weights.get(security_id).copied().unwrap_or(0.0).max(0.0),
            )
        })
        .collect();

    let total_capacity: f64 = caps.values().sum();
    let target = target_total.min(total_capacity);
    if target <= 0.0 {
        return weights
            .keys()
            .map(|security_id| (security_id.clone(), 0.0))
            .collect();
    }

    let mut result: HashMap<String, f64> = weights
        .keys()
        .map(|security_id| (security_id.clone(), 0.0))
        .collect();
    let mut active: Vec<String> = weights
        .keys()
        .filter(|security_id| caps.get(*security_id).copied().unwrap_or(0.0) > 0.0)
        .cloned()
        .collect();
    active.sort();
    let mut remaining_target = target;

    while !active.is_empty() && remaining_target > FLOAT_EPSILON {
        let total_base: f64 = active
            .iter()
            .map(|security_id| weights.get(security_id).copied().unwrap_or(0.0))
            .sum();

        let provisional: Vec<(String, f64)> = if total_base <= FLOAT_EPSILON {
            let equal = remaining_target / active.len() as f64;
            active
                .iter()
                .map(|security_id| (security_id.clone(), equal))
                .collect()
        } else {
            active
                .iter()
                .map(|security_id| {
                    let base = weights.get(security_id).copied().unwrap_or(0.0);
                    (security_id.clone(), remaining_target * (base / total_base))
                })
                .collect()
        };

        let saturated: Vec<String> = provisional
            .iter()
            .filter_map(|(security_id, proposed)| {
                let cap = caps.get(security_id).copied().unwrap_or(0.0);
                if *proposed >= cap - FLOAT_EPSILON {
                    Some(security_id.clone())
                } else {
                    None
                }
            })
            .collect();

        if saturated.is_empty() {
            for (security_id, proposed) in provisional {
                if let Some(weight) = result.get_mut(&security_id) {
                    *weight += proposed;
                }
            }
            remaining_target = 0.0;
            break;
        }

        let saturated_lookup: HashSet<String> = saturated.iter().cloned().collect();
        for security_id in &saturated {
            let cap = caps.get(security_id).copied().unwrap_or(0.0);
            if let Some(current) = result.get_mut(security_id) {
                let room = (cap - *current).max(0.0);
                *current += room;
                remaining_target = (remaining_target - room).max(0.0);
            }
        }
        active.retain(|security_id| !saturated_lookup.contains(security_id));
    }

    if remaining_target > FLOAT_EPSILON && !active.is_empty() {
        let tail = remaining_target / active.len() as f64;
        for security_id in active {
            let cap = caps.get(&security_id).copied().unwrap_or(0.0);
            if let Some(current) = result.get_mut(&security_id) {
                let room = (cap - *current).max(0.0);
                *current += room.min(tail);
            }
        }
    }

    result
}

fn constructor_blend_with_turnover_limit(
    target_weights: &HashMap<String, f64>,
    previous_weights: &HashMap<String, f64>,
    max_turnover: f64,
) -> HashMap<String, f64> {
    let normalized_max_turnover = max_turnover.clamp(0.0, 2.0);
    let mut ids: HashSet<String> = target_weights.keys().cloned().collect();
    ids.extend(previous_weights.keys().cloned());

    let turnover = 0.5
        * ids
            .iter()
            .map(|security_id| {
                let target = target_weights.get(security_id).copied().unwrap_or(0.0);
                let previous = previous_weights.get(security_id).copied().unwrap_or(0.0);
                (target - previous).abs()
            })
            .sum::<f64>();

    if turnover <= normalized_max_turnover + FLOAT_EPSILON || turnover <= FLOAT_EPSILON {
        return ids
            .into_iter()
            .map(|security_id| {
                (
                    security_id.clone(),
                    target_weights.get(&security_id).copied().unwrap_or(0.0),
                )
            })
            .collect();
    }

    let blend = normalized_max_turnover / turnover;
    ids.into_iter()
        .map(|security_id| {
            let target = target_weights.get(&security_id).copied().unwrap_or(0.0);
            let previous = previous_weights.get(&security_id).copied().unwrap_or(0.0);
            (security_id, previous + blend * (target - previous))
        })
        .collect()
}

fn normalize_constructor_country_targets(
    country_targets: Option<Vec<(String, f64)>>,
) -> Result<Vec<(String, f64)>, KernelError> {
    let normalized_targets = match country_targets {
        Some(entries) => entries,
        None => vec![
            ("JP".to_string(), 1.0 / 3.0),
            ("KR".to_string(), 1.0 / 3.0),
            ("US".to_string(), 1.0 / 3.0),
        ],
    };

    if normalized_targets.is_empty() {
        return Err(KernelError::ExecutionFailed);
    }

    let mut result: Vec<(String, f64)> = Vec::new();
    for (country, target) in normalized_targets {
        let normalized_country = country.trim().to_ascii_uppercase();
        if normalized_country.is_empty() {
            return Err(KernelError::ExecutionFailed);
        }
        if target < 0.0 {
            return Err(KernelError::ExecutionFailed);
        }
        result.push((normalized_country, target));
    }

    let target_sum: f64 = result.iter().map(|(_, weight)| *weight).sum();
    if (target_sum - 1.0).abs() > FLOAT_EPSILON {
        return Err(KernelError::ExecutionFailed);
    }

    Ok(result)
}

fn parse_previous_weights(risk_controls: &JsonMap) -> Result<HashMap<String, f64>, KernelError> {
    let Some(raw_previous) = risk_controls.get("previous_weights") else {
        return Ok(HashMap::new());
    };

    let mapping = raw_previous.as_object().ok_or(KernelError::ExecutionFailed)?;
    let mut previous_weights: HashMap<String, f64> = HashMap::new();
    for (security_id, weight_value) in mapping {
        let key = security_id.trim().to_string();
        if key.is_empty() {
            return Err(KernelError::ExecutionFailed);
        }
        let weight = to_float(weight_value)?;
        if weight < 0.0 {
            return Err(KernelError::ExecutionFailed);
        }
        previous_weights.insert(key, weight);
    }
    Ok(previous_weights)
}

fn run_constructor_core(request: ConstructorKernelRequest) -> Result<ConstructorResult, KernelError> {
    if DEFAULT_MAX_HOLDINGS == 0 {
        return Err(KernelError::ExecutionFailed);
    }
    if !(0.0 < DEFAULT_MAX_SINGLE_NAME_WEIGHT && DEFAULT_MAX_SINGLE_NAME_WEIGHT <= 1.0) {
        return Err(KernelError::ExecutionFailed);
    }
    if !(0.0..1.0).contains(&DEFAULT_COUNTRY_TOLERANCE) {
        return Err(KernelError::ExecutionFailed);
    }

    let risk_controls = request.risk_controls.unwrap_or_default();
    let te_active_l2_cap = risk_controls
        .get("te_active_l2_cap")
        .map(to_float)
        .transpose()?
        .unwrap_or(DEFAULT_TE_ACTIVE_L2_CAP);
    let alpha_tilt_strength = risk_controls
        .get("alpha_tilt_strength")
        .map(to_float)
        .transpose()?
        .unwrap_or(DEFAULT_ALPHA_TILT_STRENGTH);
    let max_adv_participation = risk_controls
        .get("max_adv_participation")
        .map(to_float)
        .transpose()?
        .unwrap_or(DEFAULT_MAX_ADV_PARTICIPATION);
    let portfolio_value_krw = risk_controls
        .get("portfolio_value_krw")
        .map(to_float)
        .transpose()?
        .unwrap_or(DEFAULT_PORTFOLIO_VALUE_KRW);
    let max_turnover = match risk_controls.get("max_turnover") {
        Some(Value::Null) => None,
        Some(value) => Some(to_float(value)?),
        None => None,
    };
    let previous_weights = parse_previous_weights(&risk_controls)?;

    if te_active_l2_cap < 0.0 {
        return Err(KernelError::ExecutionFailed);
    }
    if alpha_tilt_strength < 0.0 {
        return Err(KernelError::ExecutionFailed);
    }
    if max_adv_participation <= 0.0 {
        return Err(KernelError::ExecutionFailed);
    }
    if portfolio_value_krw <= 0.0 {
        return Err(KernelError::ExecutionFailed);
    }
    if matches!(max_turnover, Some(value) if value < 0.0) {
        return Err(KernelError::ExecutionFailed);
    }

    let normalized_targets = normalize_constructor_country_targets(request.country_targets)?;
    let mut countries: Vec<String> = normalized_targets
        .iter()
        .map(|(country, _)| country.clone())
        .collect();
    countries.sort();
    countries.dedup();
    let target_weights: HashMap<String, f64> = normalized_targets.into_iter().collect();

    let eligible_rows = parse_constructor_rows(&request.ranked_factor_rows)?;

    let mut rows_by_country: HashMap<String, Vec<ConstructorRankedRow>> = countries
        .iter()
        .map(|country| (country.clone(), Vec::new()))
        .collect();
    for row in &eligible_rows {
        if let Some(bucket) = rows_by_country.get_mut(&row.country) {
            bucket.push(row.clone());
        }
    }

    for country in &countries {
        if let Some(rows) = rows_by_country.get_mut(country) {
            rows.sort_by(|left, right| {
                left.rank_in_country
                    .unwrap_or(SENTINEL_RANK)
                    .cmp(&right.rank_in_country.unwrap_or(SENTINEL_RANK))
                    .then_with(|| right.factor_value.total_cmp(&left.factor_value))
                    .then_with(|| left.security_id.cmp(&right.security_id))
            });
        }
    }

    let mut reasons: HashSet<String> = HashSet::new();
    let requested = constructor_requested_counts(DEFAULT_MAX_HOLDINGS, &countries, &target_weights);
    let mut selected: Vec<ConstructorRankedRow> = Vec::new();
    let mut selected_country_counts: HashMap<String, usize> = countries
        .iter()
        .map(|country| (country.clone(), 0usize))
        .collect();

    for country in &countries {
        let requested_count = requested.get(country).copied().unwrap_or(0);
        let picks: Vec<ConstructorRankedRow> = rows_by_country
            .get(country)
            .map(|rows| rows.iter().take(requested_count).cloned().collect())
            .unwrap_or_default();
        selected.extend(picks.iter().cloned());
        selected_country_counts.insert(country.clone(), picks.len());
        if picks.len() < requested_count {
            reasons.insert(format!("INSUFFICIENT_NAMES_{country}"));
        }
    }

    if selected.len() < DEFAULT_MAX_HOLDINGS {
        let mut remaining_pool: Vec<ConstructorRankedRow> = Vec::new();
        for country in &countries {
            let start_index = selected_country_counts.get(country).copied().unwrap_or(0);
            if let Some(country_rows) = rows_by_country.get(country) {
                remaining_pool.extend(country_rows.iter().skip(start_index).cloned());
            }
        }
        remaining_pool.sort_by(|left, right| {
            left.rank_in_country
                .unwrap_or(SENTINEL_RANK)
                .cmp(&right.rank_in_country.unwrap_or(SENTINEL_RANK))
                .then_with(|| right.factor_value.total_cmp(&left.factor_value))
                .then_with(|| left.country.cmp(&right.country))
                .then_with(|| left.security_id.cmp(&right.security_id))
        });

        let need = DEFAULT_MAX_HOLDINGS - selected.len();
        for row in remaining_pool.into_iter().take(need) {
            if let Some(value) = selected_country_counts.get_mut(&row.country) {
                *value += 1;
            }
            selected.push(row);
        }
    }

    if selected.len() < DEFAULT_MAX_HOLDINGS {
        reasons.insert("TOTAL_UNDER_MAX_HOLDINGS".to_string());
    }

    selected.sort_by(|left, right| {
        left.country
            .cmp(&right.country)
            .then_with(|| {
                left.rank_in_country
                    .unwrap_or(SENTINEL_RANK)
                    .cmp(&right.rank_in_country.unwrap_or(SENTINEL_RANK))
            })
            .then_with(|| left.security_id.cmp(&right.security_id))
    });

    let mut country_caps: HashMap<String, f64> = HashMap::new();
    let mut country_weights: HashMap<String, f64> = countries
        .iter()
        .map(|country| (country.clone(), 0.0))
        .collect();
    for country in &countries {
        let selected_count = selected_country_counts.get(country).copied().unwrap_or(0);
        let cap = selected_count as f64 * DEFAULT_MAX_SINGLE_NAME_WEIGHT;
        country_caps.insert(country.clone(), cap);

        if selected_count == 0 {
            reasons.insert(format!("NO_SELECTED_NAMES_{country}"));
            continue;
        }

        let target = target_weights.get(country).copied().unwrap_or(0.0);
        if cap < target {
            reasons.insert(format!("COUNTRY_CAPACITY_BIND_{country}"));
        }
        country_weights.insert(country.clone(), target.min(cap));
    }

    let allocated: f64 = country_weights.values().sum();
    let mut remaining_weight = (1.0 - allocated).max(0.0);
    if remaining_weight > 0.0 {
        for country in &countries {
            if remaining_weight <= 0.0 {
                break;
            }
            let current = country_weights.get(country).copied().unwrap_or(0.0);
            let cap = country_caps.get(country).copied().unwrap_or(0.0);
            let spare = (cap - current).max(0.0);
            if spare <= 0.0 {
                continue;
            }
            let increment = spare.min(remaining_weight);
            country_weights.insert(country.clone(), current + increment);
            remaining_weight -= increment;
        }
    }

    if remaining_weight > FLOAT_EPSILON {
        reasons.insert("UNALLOCATED_CASH_DUE_TO_CAPACITY".to_string());
    }

    for country in &countries {
        let selected_count = selected_country_counts.get(country).copied().unwrap_or(0);
        if selected_count == 0 {
            continue;
        }
        let target = target_weights.get(country).copied().unwrap_or(0.0);
        let lower = (target - DEFAULT_COUNTRY_TOLERANCE).max(0.0);
        let upper = (target + DEFAULT_COUNTRY_TOLERANCE).min(1.0);
        let weight = country_weights.get(country).copied().unwrap_or(0.0);
        if weight < lower - FLOAT_EPSILON || weight > upper + FLOAT_EPSILON {
            reasons.insert(format!("COUNTRY_TOLERANCE_UNMET_{country}"));
        }
    }

    let mut rows_by_country_selected: HashMap<String, Vec<ConstructorRankedRow>> = countries
        .iter()
        .map(|country| (country.clone(), Vec::new()))
        .collect();
    for row in &selected {
        if let Some(bucket) = rows_by_country_selected.get_mut(&row.country) {
            bucket.push(row.clone());
        }
    }

    let mut final_weights: HashMap<String, f64> = HashMap::new();
    let mut benchmark_weights: HashMap<String, f64> = HashMap::new();
    let mut caps_by_security: HashMap<String, f64> = HashMap::new();

    for country in &countries {
        let country_rows = rows_by_country_selected.get(country).cloned().unwrap_or_default();
        let target_country_weight = country_weights.get(country).copied().unwrap_or(0.0);
        if country_rows.is_empty() || target_country_weight <= 0.0 {
            continue;
        }

        let mut benchmark_seed: HashMap<String, f64> = HashMap::new();
        for row in &country_rows {
            let mut seed = row
                .benchmark_proxy_weight
                .unwrap_or_else(|| row.median_traded_value_krw.unwrap_or(1.0));
            if seed <= 0.0 {
                seed = row.median_traded_value_krw.unwrap_or(1.0);
            }
            benchmark_seed.insert(row.security_id.clone(), seed.max(1.0));
        }

        let seed_sum: f64 = benchmark_seed.values().sum();
        let benchmark_country: HashMap<String, f64> = benchmark_seed
            .iter()
            .map(|(security_id, value)| (security_id.clone(), value / seed_sum))
            .collect();

        let factor_values: Vec<f64> = country_rows.iter().map(|row| row.factor_value).collect();
        let mean_factor = factor_values.iter().sum::<f64>() / factor_values.len() as f64;
        let std_factor = constructor_safe_std(&factor_values);
        let zscores: HashMap<String, f64> = country_rows
            .iter()
            .map(|row| {
                let zscore = if std_factor > FLOAT_EPSILON {
                    (row.factor_value - mean_factor) / std_factor
                } else {
                    0.0
                };
                (row.security_id.clone(), zscore)
            })
            .collect();

        let mut raw_weights: HashMap<String, f64> = HashMap::new();
        let mut country_caps_by_security: HashMap<String, f64> = HashMap::new();

        for row in &country_rows {
            let security_id = row.security_id.clone();
            let bench = benchmark_country.get(&security_id).copied().unwrap_or(0.0);
            let score = zscores.get(&security_id).copied().unwrap_or(0.0);
            let raw = (bench * (1.0 + alpha_tilt_strength * score)).max(0.0);
            raw_weights.insert(security_id.clone(), raw);

            let mut adv_cap = DEFAULT_MAX_SINGLE_NAME_WEIGHT;
            if let Some(liquidity) = row.median_traded_value_krw {
                if liquidity > 0.0 {
                    adv_cap = ((max_adv_participation * liquidity) / portfolio_value_krw)
                        .min(DEFAULT_MAX_SINGLE_NAME_WEIGHT);
                    if adv_cap < DEFAULT_MAX_SINGLE_NAME_WEIGHT - FLOAT_EPSILON {
                        reasons.insert("ADV_CAP_BIND".to_string());
                    }
                }
            }
            let cap = adv_cap.min(DEFAULT_MAX_SINGLE_NAME_WEIGHT).max(0.0);
            if cap <= 0.0 {
                reasons.insert(format!("ZERO_CAP_{country}"));
            }
            country_caps_by_security.insert(security_id.clone(), cap);
            caps_by_security.insert(security_id.clone(), cap);
            benchmark_weights.insert(security_id, target_country_weight * bench);
        }

        let country_allocated = constructor_rescale_with_caps(
            &raw_weights,
            &country_caps_by_security,
            target_country_weight,
        );
        final_weights.extend(country_allocated);
    }

    if te_active_l2_cap > 0.0 && !final_weights.is_empty() {
        let active_l2 = final_weights
            .iter()
            .map(|(security_id, weight)| {
                let benchmark_weight = benchmark_weights.get(security_id).copied().unwrap_or(0.0);
                let active = weight - benchmark_weight;
                active * active
            })
            .sum::<f64>()
            .sqrt();

        if active_l2 > te_active_l2_cap + FLOAT_EPSILON {
            reasons.insert("TE_ACTIVE_L2_CAP_BIND".to_string());
            let scale = te_active_l2_cap / active_l2;
            let mut te_adjusted_weights: HashMap<String, f64> = HashMap::new();
            for country in &countries {
                let country_rows = rows_by_country_selected.get(country).cloned().unwrap_or_default();
                if country_rows.is_empty() {
                    continue;
                }
                let target_country_weight = country_weights.get(country).copied().unwrap_or(0.0);
                let mut raw_country: HashMap<String, f64> = HashMap::new();
                let mut caps_country: HashMap<String, f64> = HashMap::new();
                for row in &country_rows {
                    let security_id = row.security_id.clone();
                    let bench = benchmark_weights.get(&security_id).copied().unwrap_or(0.0);
                    let current = final_weights.get(&security_id).copied().unwrap_or(0.0);
                    let active_component = current - bench;
                    raw_country.insert(security_id.clone(), (bench + active_component * scale).max(0.0));
                    caps_country.insert(
                        security_id.clone(),
                        caps_by_security
                            .get(&security_id)
                            .copied()
                            .unwrap_or(DEFAULT_MAX_SINGLE_NAME_WEIGHT),
                    );
                }
                te_adjusted_weights.extend(constructor_rescale_with_caps(
                    &raw_country,
                    &caps_country,
                    target_country_weight,
                ));
            }
            final_weights = te_adjusted_weights;
        }
    }

    if let Some(turnover_cap) = max_turnover {
        if turnover_cap >= 0.0 && !final_weights.is_empty() && !previous_weights.is_empty() {
            let blended = constructor_blend_with_turnover_limit(&final_weights, &previous_weights, turnover_cap);
            let turnover_bind = final_weights.iter().any(|(security_id, weight)| {
                let blended_weight = blended.get(security_id).copied().unwrap_or(0.0);
                (blended_weight - *weight).abs() > FLOAT_EPSILON
            });
            if turnover_bind {
                reasons.insert("TURNOVER_CAP_BIND".to_string());
                let mut turnover_adjusted_weights: HashMap<String, f64> = HashMap::new();
                for country in &countries {
                    let country_rows = rows_by_country_selected.get(country).cloned().unwrap_or_default();
                    if country_rows.is_empty() {
                        continue;
                    }
                    let target_country_weight = country_weights.get(country).copied().unwrap_or(0.0);
                    let raw_country: HashMap<String, f64> = country_rows
                        .iter()
                        .map(|row| {
                            (
                                row.security_id.clone(),
                                blended.get(&row.security_id).copied().unwrap_or(0.0).max(0.0),
                            )
                        })
                        .collect();
                    let caps_country: HashMap<String, f64> = country_rows
                        .iter()
                        .map(|row| {
                            (
                                row.security_id.clone(),
                                caps_by_security
                                    .get(&row.security_id)
                                    .copied()
                                    .unwrap_or(DEFAULT_MAX_SINGLE_NAME_WEIGHT),
                            )
                        })
                        .collect();
                    turnover_adjusted_weights.extend(constructor_rescale_with_caps(
                        &raw_country,
                        &caps_country,
                        target_country_weight,
                    ));
                }
                final_weights = turnover_adjusted_weights;
            }
        }
    }

    let mut holdings: Vec<ConstructorHolding> = Vec::new();
    for country in &countries {
        let country_rows = rows_by_country_selected.get(country).cloned().unwrap_or_default();
        for row in country_rows {
            let weight = final_weights.get(&row.security_id).copied().unwrap_or(0.0);
            if weight <= 0.0 {
                continue;
            }
            holdings.push(ConstructorHolding {
                security_id: row.security_id,
                country: country.clone(),
                weight,
                rank_in_country: row.rank_in_country,
                factor_value: Some(row.factor_value),
            });
        }
    }

    holdings.sort_by(|left, right| {
        left.country
            .cmp(&right.country)
            .then_with(|| {
                left.rank_in_country
                    .unwrap_or(SENTINEL_RANK)
                    .cmp(&right.rank_in_country.unwrap_or(SENTINEL_RANK))
            })
            .then_with(|| left.security_id.cmp(&right.security_id))
    });

    let mut realized_country_weights: HashMap<String, f64> = countries
        .iter()
        .map(|country| (country.clone(), 0.0))
        .collect();
    for holding in &holdings {
        if let Some(weight) = realized_country_weights.get_mut(&holding.country) {
            *weight += holding.weight;
        }
    }

    let total_weight: f64 = holdings.iter().map(|holding| holding.weight).sum();
    let mut cash_weight = (1.0 - total_weight).max(0.0);
    if cash_weight < FLOAT_EPSILON {
        cash_weight = 0.0;
    }

    let diagnostics = ConstructorSelectionDiagnostics {
        requested_holdings: DEFAULT_MAX_HOLDINGS,
        selected_holdings: holdings.len(),
        available_eligible: eligible_rows.len(),
        requested_country_counts: countries
            .iter()
            .map(|country| (country.clone(), requested.get(country).copied().unwrap_or(0)))
            .collect(),
        available_country_counts: countries
            .iter()
            .map(|country| {
                (
                    country.clone(),
                    rows_by_country.get(country).map_or(0, std::vec::Vec::len),
                )
            })
            .collect(),
        selected_country_counts: countries
            .iter()
            .map(|country| {
                (
                    country.clone(),
                    selected_country_counts.get(country).copied().unwrap_or(0),
                )
            })
            .collect(),
        country_weights: countries
            .iter()
            .map(|country| {
                (
                    country.clone(),
                    realized_country_weights.get(country).copied().unwrap_or(0.0),
                )
            })
            .collect(),
        cash_weight,
    };

    let mut fallback_reasons: Vec<String> = reasons.into_iter().collect();
    fallback_reasons.sort();

    Ok(ConstructorResult {
        holdings,
        fallback_triggered: !fallback_reasons.is_empty(),
        fallback_reasons,
        jp_odd_lot_enabled: true,
        diagnostics,
    })
}

pub fn run_ranking_kernel(request_json: &str) -> KernelResult {
    if !cfg!(feature = "kernel-ranking") {
        return Err(KernelError::NotEnabled);
    }

    let request: RankingKernelRequest =
        serde_json::from_str(request_json).map_err(|_| KernelError::InvalidRequest)?;
    let output = run_ranking_core(request).map_err(|_| KernelError::ExecutionFailed)?;
    serde_json::to_string(&output).map_err(|_| KernelError::SerializationFailed)
}

pub fn run_constructor_kernel(request_json: &str) -> KernelResult {
    if !cfg!(feature = "kernel-constructor") {
        return Err(KernelError::NotEnabled);
    }

    let request: ConstructorKernelRequest =
        serde_json::from_str(request_json).map_err(|_| KernelError::InvalidRequest)?;
    let output = run_constructor_core(request).map_err(|_| KernelError::ExecutionFailed)?;
    serde_json::to_string(&output).map_err(|_| KernelError::SerializationFailed)
}

pub fn run_backtest_kernel(request_json: &str) -> KernelResult {
    if !cfg!(feature = "kernel-backtest") {
        return Err(KernelError::NotEnabled);
    }

    let request: BacktestKernelRequest =
        serde_json::from_str(request_json).map_err(|_| KernelError::InvalidRequest)?;
    let output = run_backtest_core(request).map_err(|_| KernelError::ExecutionFailed)?;
    serde_json::to_string(&output).map_err(|_| KernelError::SerializationFailed)
}
