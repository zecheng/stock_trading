import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm
import platform

# 设置中文字体支持
def setup_chinese_font():
    """设置中文字体，根据不同操作系统选择合适的字体"""
    system = platform.system()
    
    if system == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/msyh.ttf',    # 微软雅黑
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
        ]
    elif system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/PingFang.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/Library/Fonts/Arial Unicode.ttf',
            '/System/Library/Fonts/Supplemental/Songti.ttc'
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/arphic/uming.ttc'
        ]
    
    # 尝试找到可用的中文字体
    chinese_font = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            chinese_font = FontProperties(fname=font_path)
            print(f"使用中文字体: {font_path}")
            break
    
    if chinese_font is None:
        # 如果找不到系统字体，尝试使用matplotlib内置字体
        fonts = [f.name for f in fm.fontManager.ttflist]
        for font in ['SimHei', 'Microsoft YaHei', 'STSong', 'WenQuanYi Micro Hei', 'Arial Unicode MS']:
            if font in fonts:
                chinese_font = FontProperties(family=font)
                print(f"使用内置字体: {font}")
                break
    
    if chinese_font is None:
        print("警告: 未找到合适的中文字体，图表中的中文可能显示为乱码")
        chinese_font = FontProperties()
    
    # 设置全局字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    return chinese_font

# 获取中文字体
chinese_font = setup_chinese_font()

# 中英文标签映射
labels = {
    'en': {
        'actual_price': 'Actual Price',
        'lstm_prediction': 'LSTM Prediction',
        'stock_price_prediction': 'Stock Price Prediction',
        'date': 'Date',
        'price': 'Price',
        'prediction_accuracy': 'Prediction Accuracy',
        'train_loss': 'Train Loss',
        'validation_loss': 'Validation Loss',
        'epoch': 'Epoch',
        'loss': 'Loss',
        'training_validation_loss': 'Training and Validation Loss for',
        'naive_strategy': 'Naive Strategy',
        'lstm_strategy': 'LSTM Strategy',
        'cumulative_earnings': 'Cumulative Earnings Percentages for',
        'percentage': 'Percentage (%)',
        'prediction_accuracy_across': 'Prediction Accuracy Across Stocks',
        'stock': 'Stock',
        'accuracy': 'Accuracy (%)',
        'buying_signal': 'Buying Signal',
        'selling_signal': 'Selling Signal',
        'total_gains': 'Total Gains',
        'total_investment': 'Total Investment'
    },
    'zh': {
        'actual_price': '实际价格',
        'lstm_prediction': 'LSTM预测',
        'stock_price_prediction': '股票价格预测',
        'date': '日期',
        'price': '价格',
        'prediction_accuracy': '预测准确率',
        'train_loss': '训练损失',
        'validation_loss': '验证损失',
        'epoch': '训练轮次',
        'loss': '损失值',
        'training_validation_loss': '训练与验证损失 -',
        'naive_strategy': 'Naive策略',
        'lstm_strategy': 'LSTM策略',
        'cumulative_earnings': '累积收益率 -',
        'percentage': '收益率 (%)',
        'prediction_accuracy_across': '各股票预测准确率对比',
        'stock': '股票代码',
        'accuracy': '准确率 (%)',
        'buying_signal': '买入信号',
        'selling_signal': '卖出信号',
        'total_gains': '总收益',
        'total_investment': '投资回报率'
    }
}

# 默认使用英文
current_lang = 'en'

def set_language(lang):
    """设置语言，'en'为英文，'zh'为中文"""
    global current_lang
    if lang in ['en', 'zh']:
        current_lang = lang
        print(f"已设置语言为: {'英文' if lang == 'en' else '中文'}")
    else:
        print(f"不支持的语言: {lang}，使用默认语言(英文)")

def get_label(key):
    """获取当前语言的标签"""
    return labels[current_lang].get(key, key)

def plot_stock_prediction(ticker, test_indices, actual_prices, predicted_prices, metrics, save_dir):
    """
    绘制股票预测结果对比图
    
    参数:
        ticker: 股票代码
        test_indices: 测试集日期索引
        actual_prices: 实际价格
        predicted_prices: 预测价格
        metrics: 包含rmse、mae和accuracy的字典
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    plt.figure(figsize=(15, 7))
    plt.plot(test_indices, actual_prices, label=get_label('actual_price'), color='blue', linewidth=2, alpha=0.7)
    plt.plot(test_indices, predicted_prices, label=get_label('lstm_prediction'), color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    plt.title(f'{ticker} {get_label("stock_price_prediction")}\nRMSE: {metrics["rmse"]:.2f}, MAE: {metrics["mae"]:.2f}', fontproperties=chinese_font)
    plt.xlabel(get_label('date'), fontproperties=chinese_font)
    plt.ylabel(get_label('price'), fontproperties=chinese_font)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend(prop=chinese_font)
    
    plt.text(0.02, 0.95, f'{get_label("prediction_accuracy")}: {metrics["accuracy"]*100:.2f}%', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), fontproperties=chinese_font)
    
    plt.tight_layout()
    
    prediction_dir = os.path.join(save_dir, 'pic/predictions')
    os.makedirs(prediction_dir, exist_ok=True)
    save_path = os.path.join(prediction_dir, f'{ticker}_prediction.png')
    plt.savefig(save_path, dpi=300)  # 增加DPI以提高清晰度
    plt.close()
    
    return save_path

def plot_training_loss(ticker, train_losses, val_losses, save_dir):
    """
    绘制训练和验证损失曲线
    
    参数:
        ticker: 股票代码
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label=get_label('train_loss'))
    plt.plot(val_losses, label=get_label('validation_loss'))
    plt.xlabel(get_label('epoch'), fontproperties=chinese_font)
    plt.ylabel(get_label('loss'), fontproperties=chinese_font)
    plt.title(f'{get_label("training_validation_loss")} {ticker}', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    plt.grid(True)
    
    loss_dir = os.path.join(save_dir, 'pic/loss')
    os.makedirs(loss_dir, exist_ok=True)
    save_path = os.path.join(loss_dir, f'{ticker}_loss.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def plot_cumulative_earnings(ticker, test_indices, actual_percentages, predict_percentages, save_dir):
    """
    绘制累积收益率曲线
    
    参数:
        ticker: 股票代码
        test_indices: 测试集日期索引
        actual_percentages: 实际收益率列表
        predict_percentages: 预测收益率列表
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    cumulative_naive_percentage = np.cumsum(actual_percentages)
    cumulative_lstm_percentage = np.cumsum(
        [a if p > 0 else 0 for p, a in zip(predict_percentages, actual_percentages)]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(test_indices, cumulative_naive_percentage, marker='o', markersize=3, 
             linestyle='-', color='blue', label=get_label('naive_strategy'))
    plt.plot(test_indices, cumulative_lstm_percentage, marker='o', markersize=3, 
             linestyle='-', color='orange', label=get_label('lstm_strategy'))
    plt.title(f'{get_label("cumulative_earnings")} {ticker}', fontproperties=chinese_font)
    plt.xlabel(get_label('date'), fontproperties=chinese_font)
    plt.ylabel(get_label('percentage'), fontproperties=chinese_font)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(prop=chinese_font)
    plt.tight_layout()
    
    earnings_dir = os.path.join(save_dir, 'pic/earnings')
    os.makedirs(earnings_dir, exist_ok=True)
    save_path = os.path.join(earnings_dir, f'{ticker}_cumulative.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def plot_accuracy_comparison(prediction_metrics, save_dir):
    """
    绘制所有股票预测准确度对比图
    
    参数:
        prediction_metrics: 包含每个股票预测指标的字典
        save_dir: 图片保存的根目录
    返回:
        str: 保存的图片路径
    """
    plt.figure(figsize=(15, 6))
    accuracies = [metrics['accuracy'] * 100 for metrics in prediction_metrics.values()]
    plt.bar(prediction_metrics.keys(), accuracies)
    plt.title(get_label('prediction_accuracy_across'), fontproperties=chinese_font)
    plt.xlabel(get_label('stock'), fontproperties=chinese_font)
    plt.ylabel(get_label('accuracy'), fontproperties=chinese_font)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    prediction_dir = os.path.join(save_dir, 'pic')
    os.makedirs(prediction_dir, exist_ok=True)
    save_path = os.path.join(prediction_dir, 'accuracy_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path

def plot_trading_result(ticker, close_prices, states_buy, states_sell, total_gains, invest, save_dir):
    """
    绘制交易结果图表
    
    参数:
        ticker: 股票代码
        close_prices: 收盘价列表
        states_buy: 买入点列表
        states_sell: 卖出点列表
        total_gains: 总收益
        invest: 投资回报率
        save_dir: 保存路径
    返回:
        str: 保存的图片路径
    """
    plt.figure(figsize=(15, 5))
    plt.plot(close_prices, color='r', lw=2.)
    plt.plot(close_prices, '^', markersize=10, color='m', label=get_label('buying_signal'), markevery=states_buy)
    plt.plot(close_prices, 'v', markersize=10, color='k', label=get_label('selling_signal'), markevery=states_sell)
    plt.title(f'{ticker} {get_label("total_gains")} ${total_gains:.2f}, {get_label("total_investment")} {invest:.2f}%', fontproperties=chinese_font)
    plt.legend(prop=chinese_font)
    
    # 创建保存目录
    trades_dir = os.path.join(save_dir, 'pic/trades')
    os.makedirs(trades_dir, exist_ok=True)
    
    # 保存图片
    save_path = os.path.join(trades_dir, f'{ticker}_trades.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    return save_path
