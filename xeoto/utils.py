from dataclasses import dataclass
@dataclass
class LabelGroup:
    bien_so_xe: any
    thong_bao: any
    thanh_tien: any
    thoi_gioi_vao: any

@dataclass
class DataXeVao:
    hinh_dau_xe: any
    hinh_duoi_xe: any
    bsx_dau: any
    bsx_duoi: any
    logo: any

@dataclass
class SameImage:
    same_car: bool
    same_logo: bool

@dataclass
class ListCanvas:
    bsx_dau: any
    bsx_duoi: any
    dau_xe: any
    duoi_xe: any
    logo: any

def update_label_content(text_widget, new_content, bg="gray"):
    # Mở widget để chỉnh sửa
    text_widget.config(state="normal")
    
    # Xóa content cũ (dòng 2 trở đi)
    text_widget.delete("2.0", "end")
    
    # Chèn content mới
    text_widget.insert("2.0", "\n"+new_content)
    
    # Căn giữa dòng content
    text_widget.tag_add("center", "2.0", "end")
    text_widget.tag_config("center", justify="center")
    
    # Khóa lại widget
    text_widget.config(state="disabled", bg=bg)
    text_widget.update()