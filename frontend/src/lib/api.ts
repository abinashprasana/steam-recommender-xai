export interface Metric { 0: number; 1: number }
export type Table = Record<string, Record<string, [number, number]>>;

export interface Results {
  random_split?: Table;
  temporal_split?: Table;
  protocol?: { full: Record<string, number>; sampled_99: Record<string, number> };
  parity?: { numpy: number; torch: number; torch_ci: number; agree: boolean };
  calibration_sweep?: Array<{
    lambda: number; 'NDCG@10': number; miscalibration: number; coverage: number;
  }>;
  explanations?: {
    examples: Array<{ user_idx: number; text: string; target: string;
      supporters: string[]; target_idx?: number; supporter_idx?: number[] }>;
    n_checked: number; mean_faithfulness: number; faithfulness_ci?: number;
  };
  bias?: { mean_std_entropy: number; mean_div_entropy: number; beta_sweep?: unknown[] };
}

export interface Recommendation {
  item_idx: number;
  name: string;
  genres: string[];
  explanation: string;
  influences: Array<{ name: string; rank_delta: number; score_delta: number }>;
}

export interface RecommendResponse {
  user_id: string;
  display?: string;
  history: Array<{ idx: number; name: string }>;
  recommendations: Recommendation[];
}

export interface ColdStartResponse {
  available_genres: string[];
  selected?: string[];
  games: Array<{ name: string; genres: string[]; sentiment: string; price: unknown }>;
  message?: string;
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { error?: string }).error ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  results: () => get<Results>('/api/results'),
  users: () => get<{ users: Array<{ id: string; display: string; taste: string;
    history_size: number }> }>('/api/users'),
  recommend: (id: string) => get<RecommendResponse>(`/api/recommend/${encodeURIComponent(id)}`),
  coldStart: (genres: string[]) =>
    get<ColdStartResponse>(
      '/api/cold-start' + (genres.length ? '?' + genres.map(g => `genre=${encodeURIComponent(g)}`).join('&') : ''),
    ),
  embedding: () => get<{ dims?: number; points: unknown[];
    edges?: number[]; phases?: number[] }>('/static/embedding.json'),
};

/** Colour per model, kept identical across every chart and table. */
export const SERIES: Record<string, string> = {
  Popularity: 'var(--c-popularity)',
  EASE: 'var(--c-ease)',
  BPR: 'var(--c-bpr)',
  Hybrid: 'var(--c-hybrid)',
  ItemKNN: 'var(--c-itemknn)',
  Content: 'var(--c-content)',
  Calibrated: 'var(--c-calibrated)',
};

export const seriesColor = (name: string) => SERIES[name] ?? 'var(--text-muted)';
