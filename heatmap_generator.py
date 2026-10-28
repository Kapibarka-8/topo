import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata


def create_heatmap(points, output_html='heatmap.html', grid_size=100):
    """
    points: [{'x': 100.5, 'y': 200.3, 'value': 285.5}, ...]
            координаты и значения уже в реальных единицах (метрах).
    """
    if not points:
        return None

    x = np.array([p['x'] for p in points], dtype=float)
    y = np.array([p['y'] for p in points], dtype=float)
    z = np.array([p['value'] for p in points], dtype=float)

    if len(x) < 3:
        return None

    xi = np.linspace(min(x), max(x), grid_size)
    yi = np.linspace(min(y), max(y), grid_size)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    zi = None
    for method in ('cubic', 'linear', 'nearest'):
        try:
            zi_try = griddata((x, y), z, (xi_grid, yi_grid), method=method)
            if zi_try is not None and not np.all(np.isnan(zi_try)):
                zi = zi_try
                break
        except Exception:
            continue

    if zi is None:
        return None

    if np.any(np.isnan(zi)):
        zi_nn = griddata((x, y), z, (xi_grid, yi_grid), method='nearest')
        zi = np.where(np.isnan(zi), zi_nn, zi)

    fig = go.Figure()
    fig.add_trace(go.Contour(
        z=zi, x=xi, y=yi, colorscale='Jet',
        contours=dict(showlabels=True),
        colorbar=dict(title='Высота, м'),
        name='Рельеф'
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers+text',
        marker=dict(color='black', size=6),
        text=[f'{v:.1f}' for v in z],
        textposition='top center',
        textfont=dict(size=9, color='black'),
        name='Отметки высот'
    ))
    fig.update_layout(
        title='Тепловая карта высот (оси в метрах)',
        xaxis_title='X (м)',
        yaxis_title='Y (м)'
    )
    fig.write_html(output_html)
    return output_html