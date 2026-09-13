//! Background thread sampling process/system CPU + RAM every 50 ms during the
//! benchmark, mirroring the psutil sampler in the Python scripts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use sysinfo::{Pid, System};

#[derive(Default, Clone)]
pub struct ResourceSnapshot {
    pub process_cpu: Vec<f32>,
    pub process_rss_mb: Vec<f64>,
    pub system_cpu: Vec<f32>,
    pub system_ram_percent: Vec<f64>,
}

pub struct ResourceSampler {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
    snap: Arc<Mutex<ResourceSnapshot>>,
}

impl ResourceSampler {
    pub fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let snap = Arc::new(Mutex::new(ResourceSnapshot::default()));
        let s = stop.clone();
        let sp = snap.clone();
        let handle = thread::spawn(move || {
            let mut sys = System::new_all();
            let pid = Pid::from_u32(std::process::id());
            sys.refresh_all();
            while !s.load(Ordering::Relaxed) {
                sys.refresh_all();
                let mut g = sp.lock().unwrap();
                if let Some(p) = sys.process(pid) {
                    g.process_cpu.push(p.cpu_usage());
                    g.process_rss_mb.push(p.memory() as f64 / 1048576.0);
                }
                g.system_cpu.push(sys.global_cpu_usage());
                let total = sys.total_memory();
                let used = sys.used_memory();
                g.system_ram_percent.push(if total > 0 {
                    used as f64 / total as f64 * 100.0
                } else {
                    0.0
                });
                drop(g);
                thread::sleep(Duration::from_millis(50));
            }
        });
        ResourceSampler {
            stop,
            handle: Some(handle),
            snap,
        }
    }

    pub fn snapshot(&self) -> ResourceSnapshot {
        let g = self.snap.lock().unwrap();
        ResourceSnapshot {
            process_cpu: g.process_cpu.clone(),
            process_rss_mb: g.process_rss_mb.clone(),
            system_cpu: g.system_cpu.clone(),
            system_ram_percent: g.system_ram_percent.clone(),
        }
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for ResourceSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}