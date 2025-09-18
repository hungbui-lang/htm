
# Tạo sơ đồ kiến trúc tổng quan
dot = Digraph(comment="Sơ đồ kiến trúc tổng quan", format="png")
dot.attr(rankdir="TB", size="8")

# Các node
dot.node("U", "Người dùng", shape="oval", style="filled", fillcolor="lightblue")
dot.node("I", "Giao diện nhập liệu", shape="box", style="rounded,filled", fillcolor="lightyellow")
dot.node("P", "Tiền xử lý dữ liệu", shape="box", style="rounded,filled", fillcolor="lightgrey")
dot.node("V", "Biểu diễn dữ liệu", shape="box", style="rounded,filled", fillcolor="lightgrey")
dot.node("M", "Mô hình NLP", shape="box", style="rounded,filled", fillcolor="lightpink")
dot.node("E", "Đánh giá kết quả", shape="box", style="rounded,filled", fillcolor="lightgreen")
dot.node("O", "Hiển thị bản tóm tắt", shape="box", style="rounded,filled", fillcolor="lightyellow")
dot.node("D", "Cơ sở dữ liệu lưu trữ", shape="cylinder", style="filled", fillcolor="lightblue")

# Các liên kết
dot.edges([("U", "I"), ("I", "P"), ("P", "V"), ("V", "M"), ("M", "E"), ("E", "O")])
dot.edge("O", "D")

# Xuất file
output_path = "/mnt/data/sodo_kientruc_tongquan"
dot.render(output_path, format="png", cleanup=True)

output_path + ".png"
