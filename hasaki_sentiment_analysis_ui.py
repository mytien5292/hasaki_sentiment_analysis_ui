import numpy as np
import pandas as pd
import streamlit as st
import json
import time

from hasaki_sentiment_analysis import predict_sentiment
from streamlit_searchbox import st_searchbox

# ======= Load data part =======
@st.cache_data
def load_data_products():
    data = pd.read_csv('data/san_pham_processed.csv')
    return data

@st.cache_data
def load_product_mapping():
    data = pd.read_csv('data/san_pham_processed.csv')
    product_mapping = dict(zip(data['ten_san_pham'], data['ma_san_pham']))
    return product_mapping

# ======= Logic part =======
def search_product_name(product_name):
    if "data_products" not in st.session_state:
        st.session_state.data_products = load_data_products()
    
    data_products = st.session_state.data_products
    filter_rules = data_products['ten_san_pham'].str.contains(product_name, case=False)
    product = data_products[filter_rules]

    return list(product["ten_san_pham"].values)

def search_product_code(product_code):
    if "data_products" not in st.session_state:
        st.session_state.data_products = load_data_products()
    
    data_products = st.session_state.data_products
    filter_rules = data_products['ma_san_pham'].astype(str).str.contains(product_code, case=False)
    product = data_products[filter_rules]

    return list(product["ma_san_pham"].values)

def get_product_info(product_id):
    if "data_products" not in st.session_state:
        st.session_state.data_products = load_data_products()
    
    data_products = st.session_state.data_products

    product_info = data_products[data_products['ma_san_pham'] == product_id]

    return product_info

# ======= Analysis part =======


# ======= UI part =======
def show_product_info(product_id):
    product_infos = get_product_info(product_id)

    if product_infos.empty:
        return
    
    print(product_infos)

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

def business_objective_content():
    st.image("media/hasaki_banner.jpg", width=800)
    #st.subheader("Đặt vấn đề")
    # Tiêu đề chính và các tiêu đề phụ với màu xanh lục
    st.markdown(
        """
        <h3 style='color: green;'>1. Giới thiệu bài toán Recommendation sản phẩm mỹ phẩm cho Hasaki</h3>
        """,
        unsafe_allow_html=True,
    )

    # Nội dung dưới tiêu đề chính
    st.write("""
    Trong lĩnh vực thương mại điện tử mỹ phẩm, việc cá nhân hóa trải nghiệm mua sắm là chìa khóa giúp nâng cao sự hài lòng của khách hàng và tối ưu doanh thu. Với danh mục sản phẩm đa dạng từ chăm sóc da, trang điểm đến dưỡng tóc, **Hasaki.vn** cần một hệ thống gợi ý sản phẩm thông minh để hỗ trợ khách hàng tìm kiếm và lựa chọn sản phẩm phù hợp.
    """)

    # Tiêu đề "2. Mục tiêu của hệ thống Recommendation:" với màu xanh lục
    st.markdown(
        """
        <h3 style='color: green;'>2. Mục tiêu của hệ thống Recommendation</h3>
        """,
        unsafe_allow_html=True,
    )

    # Nội dung dưới tiêu đề phụ
    st.write("""
    - **Cá nhân hóa**: Đề xuất sản phẩm dựa trên sở thích và hành vi của khách hàng.
    - **Tăng tỷ lệ chuyển đổi**: Gợi ý sản phẩm liên quan và thúc đẩy bán chéo.
    - **Độ chính xác cao**: Ứng dụng các phương pháp như Collaborative Filtering, Content-Based Filtering, và Hybrid.
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

def build_product_analysis():
    if "product_mapping" not in st.session_state:
        st.session_state.product_mapping = load_product_mapping()
    
    product_mapping = st.session_state.product_mapping

    search_type = st.radio("Chọn cách tìm kiếm sản phẩm:", ("Tìm kiếm theo tên sản phẩm", "Tìm kiếm theo mã sản phẩm"))

    if search_type == "Tìm kiếm theo tên sản phẩm":
        selected_value = st_searchbox(
            search_product_name,
            placeholder="Tìm kiếm tên sản phẩm",
        )

        selected_value = product_mapping.get(selected_value, "Không tìm thấy mã sản phẩm")
    else:
        selected_value = st_searchbox(
            search_product_code,
            placeholder="Tìm kiếm mã sản phẩm",
        )
    
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
            

# ======= Main content =======
def main_content():
    # Tiêu đề với màu xanh lục
    st.markdown(
        """
        <h1 style='color: green;'>Phân tích đánh giá sản phẩm Hasaki</h1>
        """,
        unsafe_allow_html=True,
    )
    #st.subheader("Thực hiện dự án")
###

 ###
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
    menu = ["Đặt vấn đề và thực hiện dự án", "Phân tích sản phẩm", "Phân tích dữ liệu mới"]
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

    if choice == 'Đặt vấn đề và thực hiện dự án':
        business_objective_content()
    elif choice == 'Phân tích sản phẩm':
        build_product_analysis()
    elif choice == 'Phân tích dữ liệu mới':
        new_product_analysis()