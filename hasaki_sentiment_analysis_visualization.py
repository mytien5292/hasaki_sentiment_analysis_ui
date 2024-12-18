import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from wordcloud import WordCloud
from hasaki_sentiment_analysis_prediction import VIETNAMESE_STOPWORDS_LIST

def show_overview(product_infos, product_feedbacks):
    # === Đếm số lượng feedback và vẽ piechart ===
    #st.write(f"Số lượng feedback: {len(product_feedbacks)}")
    st.markdown(
    f"<h4 style='font-weight: bold;'>Số lượng feedback: {len(product_feedbacks)}</h4>",
    unsafe_allow_html=True,
)

    # === Thống kê theo số lượng so_sao từ 5 đến 1, nếu không có thì mặc định là 0 ===
    star_counts = product_feedbacks["so_sao"].value_counts().reindex([5, 4, 3, 2, 1], fill_value=0)

    # Vẽ bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    star_counts.plot(kind='bar', ax=ax, color='blue')
    ax.set_xlabel('Số sao')
    ax.set_ylabel('Số lượng')
    ax.set_title('Số lượng feedback theo số sao')
    ax.grid(True)

    # Hiển thị bar chart bằng streamlit
    st.pyplot(fig)
    
    # === Vẽ piechart thể hiện phân phối sentiment_label ===
    # Đếm số lượng feedback theo cột sentiment_label
    feedback_counts = product_feedbacks["sentiment_label"].value_counts()

    # Vẽ piechart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(feedback_counts, labels=feedback_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.legend()

    # Hiển thị piechart bằng streamlit
    st.pyplot(fig)

    # === Vẽ piechart cho các topics theo từng sentiment_label ===
    for label in ["positive", "negative"]:
        #st.write(f"Phân phối topics cho sentiment: {label}")
        st.markdown(
    f"<h4 style='font-weight: bold;'>Phân phối topics cho sentiment: {label}</h4>",
    unsafe_allow_html=True,
)
        # Lọc các feedback theo sentiment_label hiện tại
        feedbacks = product_feedbacks[product_feedbacks["sentiment_label"] == label]

        if len(feedbacks) == 0:
            #st.write(f"Không có feedback cho sentiment: {label}")
            st.markdown(
    f"<h4> => Không có feedback cho sentiment {label}</h4>",
    unsafe_allow_html=True,
)
            continue
        
        # Đếm số lượng feedback theo cột topic
        topic_counts = feedbacks["topics"].value_counts()

        # Thay thế "No label" bằng "others"
        topic_counts.index = topic_counts.index.str.replace("No label", "others")
        
        # Vẽ piechart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.pie(topic_counts, labels=None, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.legend(topic_counts.index, title="Topics", loc="lower left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Hiển thị piechart bằng streamlit
        st.pyplot(fig)

def show_feedback_count(product_infos, product_feedbacks):
    # --- Vẽ biểu đồ thể hiện số lượng feedback theo tháng ---
    # Chuyển cột ngay_binh_luan sang kiểu datetime
    product_feedbacks['ngay_binh_luan'] = pd.to_datetime(product_feedbacks['ngay_binh_luan'], format='%d/%m/%Y')

    # Thêm cột month_year để lưu trữ tháng và năm của ngay_binh_luan
    product_feedbacks['month_year'] = product_feedbacks['ngay_binh_luan'].dt.to_period('M')

    # Đếm số lượng feedback theo từng tháng
    monthly_feedback_counts = product_feedbacks['month_year'].value_counts().sort_index()

    # Tạo một khoảng thời gian từ tháng đầu tiên đến tháng cuối cùng
    all_months = pd.period_range(start=monthly_feedback_counts.index.min(), end=monthly_feedback_counts.index.max(), freq='M')

    # Đảm bảo tất cả các tháng đều có dữ liệu, nếu không thì mặc định là 0
    monthly_feedback_counts = monthly_feedback_counts.reindex(all_months, fill_value=0)

    # Vẽ biểu đồ số lượng feedback theo từng tháng bằng linechart màu đỏ
    fig, ax = plt.subplots(figsize=(10, 6))
    monthly_feedback_counts.plot(kind='bar', ax=ax, color='red')
    ax.set_xlabel('Tháng')
    ax.set_ylabel('Số lượng feedback')
    ax.set_title('Số lượng feedback theo từng tháng')
    ax.grid(True)

    # Hiển thị biểu đồ bằng streamlit
    st.pyplot(fig)

    # --- Vẽ biểu đồ thể hiện số lượng feedback theo giờ ---
    # Chuyển cột gio_binh_luan sang kiểu datetime
    product_feedbacks['gio_binh_luan'] = product_feedbacks['gio_binh_luan'].str.replace(' ', '')
    product_feedbacks['gio_binh_luan'] = pd.to_datetime(product_feedbacks['gio_binh_luan'], format='%H:%M').dt.hour

    # Đếm số lượng feedback theo từng giờ
    hourly_feedback_counts = product_feedbacks['gio_binh_luan'].value_counts().sort_index()

    # Đảm bảo tất cả các giờ đều có dữ liệu, nếu không thì mặc định là 0
    all_hours = range(24)
    hourly_feedback_counts = hourly_feedback_counts.reindex(all_hours, fill_value=0)

    # Vẽ biểu đồ số lượng feedback theo từng giờ bằng bar chart màu xanh lá cây
    fig, ax = plt.subplots(figsize=(10, 6))
    hourly_feedback_counts.plot(kind='bar', ax=ax, color='green')
    ax.set_xlabel('Giờ')
    ax.set_ylabel('Số lượng feedback')
    ax.set_title('Số lượng feedback theo từng giờ')
    ax.set_xticks(all_hours)
    ax.set_xticklabels([f'{hour}:00' for hour in all_hours])
    ax.grid(True)

    # Hiển thị biểu đồ bằng streamlit
    st.pyplot(fig)

def show_word_cloud(product_infos, product_feedbacks):
    # --- Vẽ word cloud cho từng nhãn sentiment ---
    for label in ["positive", "negative"]:
        try:
            #st.write(f"Word Cloud cho nhãn sentiment: {label}")
            st.markdown(
    f"<h4 style='font-weight: bold;'>Word Cloud cho nhãn sentiment: {label}</h4>",
    unsafe_allow_html=True,
)
            
            # Lấy tất cả các feedback cho nhãn sentiment hiện tại
            feedbacks = ' '.join(product_feedbacks[product_feedbacks['sentiment_label'] == label]['normalized_text_with_boost_words'].values)
            
            # Tạo word cloud
            wordcloud = WordCloud(width=800, 
                                height=400, 
                                background_color='white').generate(feedbacks)
            
            # Lấy danh sách từ và trọng số
            words = wordcloud.words_

            # Xóa các từ không mong muốn
            filtered_words = {word: weight for word, weight in words.items() if word not in VIETNAMESE_STOPWORDS_LIST}

            # Tạo lại WordCloud sau khi xóa từ
            wordcloud_filtered = WordCloud(
                width=800,
                height=400,
                background_color='white'
            ).generate_from_frequencies(filtered_words)

            # Vẽ word cloud
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(wordcloud_filtered, interpolation='bilinear')
            ax.axis('off')
            
            # Hiển thị word cloud bằng streamlit
            st.pyplot(fig)
        except:
            continue

def analyze_and_visualize(product_infos, product_feedbacks):
    if len(product_feedbacks) == 0:
        #st.write("Không có dữ liệu feedback để phân tích")
        st.markdown(
    f"<h4>Không có dữ liệu feedback để phân tích</h4>",
    unsafe_allow_html=True,
)
        return

    show_overview(product_infos, product_feedbacks)
    show_feedback_count(product_infos, product_feedbacks)
    show_word_cloud(product_infos, product_feedbacks)