macro_rules! trace {
    ($($tt:tt)*) => {
        #[cfg(feature = "tracing")] {
            tracing::trace!("{:?}: {}", std::thread::current().id(), format_args!($($tt)*))
        }
    }
}

pub(crate) use trace;
