import numpy as np
import pandas as pd
import streamlit as st
import json
import time


from hasaki_sentiment_analysis_prediction import predict_sentiment
from hasaki_sentiment_analysis_visualization import analyze_and_visualize
from streamlit_searchbox import st_searchbox

# ======= Load data part =======
@st.cache_data
def load_boost_words():
    with open('data/tools/boost_words.txt', 'r', encoding='utf-8') as file:
        boost_words = [line.strip() for line in file]
    return boost_words

@st.cache_data
def load_data_products():
    data = pd.read_csv('data/san_pham_processed.csv')

    if "data_feedbacks" not in st.session_state:
        st.session_state.data_feedbacks = load_data_feedbacks()

    data_feedbacks = st.session_state.data_feedbacks

    data['so_luong_danh_gia'] = data['ma_san_pham'].map(data_feedbacks['ma_san_pham'].value_counts()).fillna(0).astype(int)

    data['ten_san_pham_sl_danh_gia'] = data['ten_san_pham'] + " (" + data['so_luong_danh_gia'].astype(str) + " đánh giá)"
    data['ma_san_pham_sl_danh_gia'] = data['ma_san_pham'].astype(str) + " (" + data['so_luong_danh_gia'].astype(str) + " đánh giá)"

    return data

@st.cache_data
def load_product_mapping():
    if "data_products" not in st.session_state:
        st.session_state.data_products = load_data_products()
    
    data = st.session_state.data_products

    product_mapping = dict(zip(data['ten_san_pham_sl_danh_gia'], data['ma_san_pham']))
    return product_mapping

def is_existed(addded_words, word):
    for x in addded_words:
        if word in x:
            return True
    return False

def apply_boost_words(text, boost_words):
    parts = text.split()
    added_words = []

    for i in range(5, 0, -1):
        for j in range(len(parts) - i + 1):
            sub_text = ' '.join(parts[j:j+i])
            if sub_text in boost_words and not is_existed(added_words, sub_text):
                added_words.append(sub_text)

    for word in added_words:
        num_boost = word.count(' ')
        text = text.replace(word, word.replace(" ", "_"))
        for i in range(num_boost):
            text = text + " " + word.replace(" ", "_")

    return text

@st.cache_data
def load_data_feedbacks():
    data = pd.read_csv('data/Danh_gia_with_label.csv')
    boost_words = load_boost_words()

    data['normalized_text_with_boost_words'] = data['normalized_text'].apply(lambda x: apply_boost_words(x, boost_words))

    return data

# ======= Logic part =======
def search_product_name(product_name):
    if "data_products" not in st.session_state:
        st.session_state.data_products = load_data_products()
    data_products = st.session_state.data_products
    filter_rules = data_products['ten_san_pham_sl_danh_gia'].str.contains(product_name, case=False)
    product = data_products[filter_rules]

    search_all_text = "Tìm tất cả sản phẩm có chứa từ khóa " + product_name
    result = [search_all_text] + list(product["ten_san_pham_sl_danh_gia"].values)

    return result

def search_product_code(product_code):
    if "data_products" not in st.session_state:
        st.session_state.data_products = load_data_products()
    
    data_products = st.session_state.data_products
    filter_rules = data_products['ma_san_pham_sl_danh_gia'].astype(str).str.contains(product_code, case=False)
    product = data_products[filter_rules]

    search_all_text = "Tìm tất cả sản phẩm có chứa từ khóa " + product_code
    result = [search_all_text] + list(product["ma_san_pham_sl_danh_gia"].values)

    return result

def get_product_info(product_id):
    if "data_products" not in st.session_state:
        st.session_state.data_products = load_data_products()

    if "data_feedbacks" not in st.session_state:
        st.session_state.data_feedbacks = load_data_feedbacks()
    
    data_products = st.session_state.data_products
    data_feedbacks = st.session_state.data_feedbacks

    product_feedbacks = data_feedbacks[data_feedbacks['ma_san_pham'] == product_id]
    product_info = data_products[data_products['ma_san_pham'] == product_id]

    return product_info, product_feedbacks

# ======= Analysis part =======


# ======= UI part =======
def show_product_info(product_id):
    product_infos, product_feedbacks = get_product_info(product_id)

    if product_infos.empty:
        st.write("Không tìm thấy sản phẩm.")
        return

    for product_info in product_infos.itertuples():
        #st.write(f"""##### {product_info.ten_san_pham}\n""")
        st.markdown(
            f"""
            <h5 style='color: green;'>{product_info.ten_san_pham}</h5>
            """,
            unsafe_allow_html=True,
        )
        st.image(product_info.hinh_san_pham, width=400)
        st.write(f"""[Xem chi tiết sản phẩm]({product_info.link_san_pham})""")
        
        formatted_price = f"{product_info.gia_ban:,}đ"
        st.markdown(
            f"""
            **Mã sản phẩm**: {product_info.ma_san_pham}  
            **Giá bán**: <span style='color: red;'>{formatted_price}</span>

            **Điểm trung bình**: {product_info.diem_trung_binh} ⭐
            """,
            unsafe_allow_html=True,
        )
 
    st.markdown(
        """
        <div style='background-color: #66BB6A; padding: 10px; border-radius: 5px; text-align: center;'>
            <h2 style='color: white; margin: 0;'>Phân tích sản phẩm</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    #st.image("media/sentiment_distribution.png", use_container_width=True, width=200)
    # Thêm khoảng cách trước hàng ảnh
    st.markdown("<br>", unsafe_allow_html=True)

    analyze_and_visualize(product_infos, product_feedbacks)

def business_objective_content():
    #st.image("media/hasaki_banner.jpg", width=800)

    #st.subheader("Đặt vấn đề")
    # Tiêu đề chính và các tiêu đề phụ với màu xanh lục
    st.markdown(
        """
        <h3 style='color: green;'>1. Giới thiệu bài toán Sentiment Analysis sản phẩm mỹ phẩm cho Hasaki</h3>
        """,
        unsafe_allow_html=True,
    )

    # Nội dung dưới tiêu đề chính
    st.write("""
    Hasaki.vn là một là một nền tảng bán lẻ mỹ phẩm trực tuyến lớn, có hàng ngàn sản phẩm và đánh giá từ khách hàng. Các đánh giá (reviews) này chứa thông tin quan trọng, là chìa khoá giúp Hasaki hiểu được cảm nhận của khách hàng về sản phẩm, dịch vụ và trải nghiệm mua sắm. Tuy nhiên, việc phân tích thủ công rất tốn thời gian và khó thực hiện ở quy mô lớn. Vì vậy cần phải xây dựng hệ thống phân tích cảm xúc (Sentiment Analysis) để tự động hóa việc phân loại và trích xuất thông tin từ đánh giá của khách hàng.
    """)

    # Tiêu đề "2. Mục tiêu của hệ thống Sentiment Analysis:" với màu xanh lục
    st.markdown(
        """
        <h3 style='color: green;'>2. Mục tiêu của hệ thống Sentiment Analysis</h3>
        """,
        unsafe_allow_html=True,
    )

    # Nội dung dưới tiêu đề phụ
    st.write("""
    - **Phân loại cảm xúc của các bình luận**: Xác định cảm xúc của khách hàng trong các đánh giá sản phẩm là tích cực, tiêu cực.
    - **Hiểu sâu hơn về khách hàng**: Trích xuất thông tin hữu ích như: khách hàng thích điều gì và không hài lòng điều gì.
    - **Hỗ trợ ra quyết định**: Giúp Hasaki.vn cải thiện sản phẩm và dịch vụ, tăng mức độ hài lòng và giữ chân khách hàng.
    - **Ứng dụng thực tế**: Tự động gắn nhãn đánh giá trên website. Hỗ trợ cho hệ thống chăm sóc khách hàng và nâng cao chất lượng marketing.
    """)

    # Tiêu đề "3. Lợi ích cho Hasaki:" với màu xanh lục
    st.markdown(
        """
        <h3 style='color: green;'>3. Lợi ích cho Hasaki</h3>
        """,
        unsafe_allow_html=True,
    )

    # Nội dung dưới tiêu đề phụ
    st.write("""
    - Cải thiện trải nghiệm khách hàng.
    - Tối ưu hóa chiến lược kinh doanh.
    - Khẳng định vị thế dẫn đầu trong ngành mỹ phẩm tại Việt Nam.
    """)

def build_project_construction():
    #st.image("media/hasaki_banner.jpg", width=800)
    #1. Crawl thêm dữ liệu từ các categorical
    st.markdown(
        """
        <h3 style='color: green;'>1. Crawl thêm dữ liệu từ các categorical</h3>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        Crawl thêm dữ liệu từ các categorical trên trang web Hasaki.vn: chăm sóc cơ thể, chăm sóc tóc và da đầu, trang điểm, chăm sóc cá nhân, chăm sóc da mặt.
        """,
        unsafe_allow_html=True,
    )
    st.image("media/crawl_du_lieu.png", use_container_width=True, width=800)

    #2. Labeling data 
    st.markdown(
        """
        <h3 style='color: green;'>2. Labeling data</h3>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        **Cần 2 lớp labels:**
        - 2 sắc thái của bình luận: Positive/Negative (Không có neutral do khó nhận dạng loại này)
        - 4 chủ đề (topics) mà các bình luận hay nhắc đến:
            + Pricing - Giá cả
            + Fragrance - Mùi thơm
            + Usage Experience - Trải nghiệm sử dụng
            + Body Impact - Tác động thế nào lên cơ thể
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Với label sắc thái của bình luận:**
        - Nếu review có số sao >= 4 => Positive
        - Nếu review có số sao < 4 => Negative

        Với label về chủ đề mà bình luận đang nói đến => Dùng matching từ khoá.
        """,
        unsafe_allow_html=True,
    )
    #3. Tiền xử lý dữ liệu
    st.markdown(
        """
        <h3 style='color: green;'>3. Tiền xử lý dữ liệu</h3>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **Các bước tiền xử lý dữ liệu:**
        1. Bỏ các bình luận bị duplicate hoặc nan
        2. Bỏ các dấu space, khoảng trắng dư thừa
        3. Thay thế kí tự emoji
        4. Thay thế các từ tiếng Anh thành tiếng Việt
        5. Thay thế các từ teencode thành từ đọc được
        6. Bỏ các stopword
        """,
        unsafe_allow_html=True,
    )

    #4. Phân tích dữ liệu
    st.markdown(
        """
        <h3 style='color: green;'>4. Phân tích dữ liệu</h3>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
    """
    <h4>Trực quan hoá dữ liệu sau khi phân tích bằng các biểu đồ</h4>
    """,
    unsafe_allow_html=True,
    )
    
    st.markdown(
    """
    **Tỷ lệ bình luận tích cực và tiêu cực**
    """,
    unsafe_allow_html=True,
    )
    st.image("media/ti_le_pos_neg.jpg", use_container_width=True, width=350)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    **Tỷ lệ Topics trong bình luận tích cực**
    """,
    unsafe_allow_html=True,
    )
    
    st.image("media/sentiment_label_positive.png", use_container_width=True, width=200)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    **Tỷ lệ Topics trong bình luận tiêu cực**
    """,
    unsafe_allow_html=True,
    )
    
    st.image("media/sentimet_label_negative.png", use_container_width=True, width=200)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    **Phân bố số lượng bình luận trên mỗi sản phẩm**
    """,
    unsafe_allow_html=True,
    )
    
    st.image("media/so_luong_binh_luan_tren_moi_user.png", use_container_width=True, width=800)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    **Số lượng bình luận theo tháng và năm**
    """,
    unsafe_allow_html=True,
    )
    st.image("media/so_luong_binh_luan_theo_thang_va_nam.png", use_container_width=True, width=800)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    **Số lượng bình luận theo giờ**
    """,
    unsafe_allow_html=True,
    )
    st.image("media/so_luong_binh_luan_theo_gio.png", use_container_width=True, width=800)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    **Biểu đồ tần suất số sao**
    """,
    unsafe_allow_html=True,
    )
    st.image("media/tan_suat_so_sao.png", use_container_width=True, width=800)

    st.markdown(
    """
    **WordCloud cho các sentiment labels**
    """,
    unsafe_allow_html=True,
    )
    # Tạo hai cột
    col1, col2 = st.columns(2)

    # Hiển thị hình ảnh trong cột đầu tiên
    with col1:
        st.image("media/positive_comment.png", use_container_width=True)

    # Hiển thị hình ảnh trong cột thứ hai
    with col2:
        st.image("media/negative_comment.png", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    #5. Xây dựng model
    st.markdown(
        """
        <h3 style='color: green;'>5 Xây dựng model</h3>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
    """
    <h4>5.1 Xây dựng model bằng Scikit-Learn</h4>
    """,
    unsafe_allow_html=True,
    )

    st.markdown(
        """
        - Sử dụng TF-IDF để vectorize nội dung bình luận
        - Build model bằng các thuật toán sau và kết quả
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("media/scikit_learn.png", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <h4>5.2 Xây dựng model bằng PySpark</h4>
    """,
    unsafe_allow_html=True,
    )

    st.markdown(
        """
        - Sử dụng TF-IDF để vectorize nội dung bình luận
        - Build model bằng các thuật toán sau và kết quả
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("media/PySpark.png", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
    """
    <h4>5.3 Xây dựng model bằng thư viện khác</h4>
    """,
    unsafe_allow_html=True,
    )
    st.markdown(
        """
        - Sử dụng SentenceTransformer để vectorize nội dung bình luận
        - Build model bằng các thuật toán
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("media/thu_vien_khac.png", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)


    #6. Kết luận
    st.markdown(
        """
        <h3 style='color: green;'>6 Kết luận</h3>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    Sử dụng thuật toán Random Forest để áp dụng phân loại cảm xúc của bình luận vì độ chính xác cao.
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    Nâng cao hiệu suất cho mô hình.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.image("media/nang_cao_hieu_suat_random.png", use_container_width=True)

##

def build_product_analysis():
    if "product_mapping" not in st.session_state:
        st.session_state.product_mapping = load_product_mapping()
    
    product_mapping = st.session_state.product_mapping

    search_type = st.radio("Chọn cách tìm kiếm sản phẩm:", ("Tìm kiếm theo tên sản phẩm", "Tìm kiếm theo mã sản phẩm"))

    if search_type == "Tìm kiếm theo tên sản phẩm":
        selected_value = st_searchbox(
            search_product_name,
            placeholder="Tìm kiếm tên sản phẩm",
            default_use_searchterm=True,
        )

        print("--------------------------------")
        print(selected_value)
        print("--------------------------------")

        selected_value = product_mapping.get(selected_value, "Không tìm thấy mã sản phẩm")
    else:
        selected_value = st_searchbox(
            search_product_code,
            placeholder="Tìm kiếm mã sản phẩm",
            default_use_searchterm=True,
        )

        print("--------------------------------")
        print(selected_value)
        print("--------------------------------")

        if selected_value is not None:
            selected_value = int(selected_value.split(" ")[0])
    
    show_product_info(selected_value)
   
def new_product_analysis():
    input_type = st.radio("Chọn cách nhập dữ liệu:", ("Nhập từ bàn phím", "Nhập từ file"))

    input_feedbacks = []

    if input_type == "Nhập từ bàn phím":
        feedback_content = st.text_area("Nội dung bình luận", height=200)
        st.write("""Ví dụ:\n\n- sp rất ok\n\n- Mùi hương không được thơm.\n\n- Mùi hắc cồn, bôi vào da thấy rát, cảm giác sưng sưng ở vùng bôi nách sau 2 ngày bong vảy.\n\n- Tiếc là không chịu mua em nó sớm hơn. Mình mới dùng 1 lần sau khi tắm, qua hôm sau thấy khô thoáng mà hết mùi hẳn. Đúng chân ái. Chắc phải mua 1-2 lọ trữ sẵn\n\n
                 """)
        input_feedbacks = feedback_content.split('\n')
    else:
        uploaded_file = st.file_uploader("Chọn file dữ liệu mới (csv hoặc txt)", type=["csv", "txt"], accept_multiple_files=False)

        if uploaded_file is None:
            st.write("Ví dụ file dữ liệu csv:")
            example_input_csv = pd.read_csv("data/input_file_example.csv")
            st.write(example_input_csv.head())

            st.write("Ví dụ file dữ liệu txt:")
            example_input_txt = pd.read_csv("data/input_file_example.txt", sep="\t", header=None)
            st.write(example_input_txt)

        if uploaded_file is not None:
            st.write("Nội dung dữ liệu vừa tải lên:")

            if uploaded_file.type == "text/plain":
                input_feedbacks = uploaded_file.read().decode("utf-8").splitlines()
                for feedback in input_feedbacks:
                    st.write(feedback)
            else:
                data = pd.read_csv(uploaded_file)
                st.write(data.head())
                input_feedbacks = data["noi_dung_binh_luan"].tolist()
    
    input_feedbacks = [feedback for feedback in input_feedbacks if feedback.strip()]

    if st.button("Phân tích dữ liệu"):
        st.write("Kết quả phân tích dữ liệu:")
        result = predict_sentiment(input_feedbacks)
        st.write(result)

        st.download_button(
            label="Download kết quả (.csv)",
            data=result.to_csv(index=False),
            file_name="sentiment_result.csv",
            mime="text/csv",
        )
            

# ======= Main content =======
def main_content():
    # Tiêu đề với màu xanh lục
    # Đặt cấu hình trang rộng hơn
    st.set_page_config(
        #page_title="My App",  # Tiêu đề của ứng dụng
        #page_icon="🌟",       # Biểu tượng hiển thị trên tab
        layout="wide",        # Chế độ hiển thị: "wide" hoặc "centered"
    )
    
# Hiển thị tiêu đề với màu chữ trắng và khung nền xanh lá
    # Hiển thị tiêu đề với khung nền
    st.image("media/tieu_de.png", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    #st.subheader("Thực hiện dự án")
       # Tiêu đề Menu
    st.sidebar.markdown(
        """
        <div style='color: green; font-size: 18px; font-weight: bold; margin-bottom: -20px;'>
            Menu
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Menu chính
    menu = ["Mục tiêu dự án", "Thực hiện dự án", "Phân tích sản phẩm", "Phân tích dữ liệu mới"]
    choice = st.sidebar.selectbox("", menu)

    # Tiêu đề Thành viên thực hiện
    st.sidebar.markdown("""
    <div style="color: green; font-size: 16px; font-weight: bold; margin-top: 20px;">
        Thành viên thực hiện:
    </div>
    """, unsafe_allow_html=True)

    # Hiển thị ảnh Nguyễn Thị Mỷ Tiên
    st.sidebar.image("media/tien.jpg", width=150, caption="Nguyễn Thị Mỷ Tiên", use_container_width=False)

    # Hiển thị ảnh Đặng Thị Thảo
    st.sidebar.image("media/thao.jpg", width=150, caption="Đặng Thị Thảo", use_container_width=False)
    
    # Giảng viên hướng dẫn
    st.sidebar.markdown("""
    <div style="color: green; font-size: 16px; font-weight: bold; margin-top: 20px;">
        Giảng viên hướng dẫn:
    </div>
    """, unsafe_allow_html=True)
    # Hiển thị ảnh giảng viên
    st.sidebar.image("media/co_phuong.jpg", width=150, caption="Cô Khuất Thuỳ Phương", use_container_width=False)
    
    # Thời gian thực hiện
    st.sidebar.markdown("""
    <div style="color: green; font-size: 16px; font-weight: bold; margin-top: 20px;">
        Thời gian thực hiện:
    </div>
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9; margin-top: 5px;">
        16/12/2024
    </div>
    """, unsafe_allow_html=True) 

    if choice == 'Mục tiêu dự án':
        business_objective_content()
    elif choice == 'Thực hiện dự án':
        build_project_construction()
    elif choice == 'Phân tích sản phẩm':
        build_product_analysis()
    elif choice == 'Phân tích dữ liệu mới':
        new_product_analysis()