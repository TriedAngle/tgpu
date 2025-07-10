use crate::raw::DeviceImpl;

pub enum Label<'a> {
    Named(&'a str),
    Tag(&'a [u8]),
    Both((&'a str, &'a [u8])),
}

impl DeviceImpl { 
    pub unsafe fn attach_label(&self, label: &Label<'_>) {
        // TODO
    }
}
