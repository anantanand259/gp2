/* ═══════════════════════════════════════════════════════════════
   Government Polytechnic Adityapur – JavaScript v2
   Bug-free · Smooth transitions · Professional animations
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ─────────────────────────────────────────────────────────────
// DATA (100% from GpaServer.java)
// ─────────────────────────────────────────────────────────────
const DATA = {
    stats: {
        totalStudents: 180,
        departments: 4,
        placementRate: 90,
        highestPackage: "7 LPA",
        established: 1980,
        passoutStudents: 8300,
        facultyCount: 11,
        hostelCapacity: 100
    },

    departments: [
        {
            name: "Computer Science & Engineering",
            code: "cse",
            seats: 45,
            duration: "3 Years",
            description: "Learn programming, data structures, database management, networking, and emerging technologies.",
            subjects: ["C/C++", "Java", "DBMS", "Networking", "Web Tech", "Data Structures", "Operating Systems"]
        },
        {
            name: "Mechanical Engineering",
            code: "mech",
            seats: 45,
            duration: "3 Years",
            description: "Master manufacturing, thermodynamics, machine design, and production engineering.",
            subjects: ["CAD/CAM", "Thermal Engineering", "Manufacturing", "Machine Design", "Workshop", "Fluid Mechanics"]
        },
        {
            name: "Electrical Engineering",
            code: "ee",
            seats: 45,
            duration: "3 Years",
            description: "Study power systems, control electronics, electrical machines, and power electronics.",
            subjects: ["Power Systems", "Control", "Machines", "Electronics", "Wiring", "Measurement"]
        },
        {
            name: "Metallurgical Engineering",
            code: "met",
            seats: 45,
            duration: "3 Years",
            description: "Explore materials science, iron & steel technology, heat treatment, and quality testing.",
            subjects: ["Materials", "Iron & Steel", "Heat Treatment", "Testing", "Metallurgy", "Quality Control"]
        }
    ],

    faculty: [
        // Electrical Engineering
        { name: "Mr. Bikram Majhi", designation: "Lecturer", department: "Electrical Engineering", deptCode: "ee", qualification: "B.Sc. Engg. (Electrical)", teachingExperience: "16 Years", industrialExperience: "0 Years", specialization: "Control Systems, Power Electronics" },
        { name: "Dr. Chetna Sumedha", designation: "Lecturer", department: "Electrical Engineering", deptCode: "ee", qualification: "Ph.D.", teachingExperience: "3 Years", industrialExperience: "0 Years", specialization: "Signal Processing, Control Engineering" },
        { name: "Mr. Gulshan Kalundia", designation: "Lecturer", department: "Electrical Engineering", deptCode: "ee", qualification: "B.Tech", teachingExperience: "9 Years", industrialExperience: "0 Years", specialization: "Electrical Circuits, Measurement" },
        // Mechanical Engineering
        { name: "Mr. Suresh Tiwary", designation: "Lecturer", department: "Mechanical Engineering", deptCode: "mech", qualification: "Ph.D. (Pursuing)", teachingExperience: "16 Years", industrialExperience: "4 Years", specialization: "Thermodynamics, Heat Engines, Fluid Mechanics" },
        { name: "Mr. Gulshan Kumar", designation: "Lecturer", department: "Mechanical Engineering", deptCode: "mech", qualification: "M.Tech.", teachingExperience: "10 Years", industrialExperience: "0 Years", specialization: "Machine Design, Manufacturing" },
        { name: "Md. Sakil", designation: "Lecturer", department: "Mechanical Engineering", deptCode: "mech", qualification: "B.E. / B.Tech (Mechanical)", teachingExperience: "5 Years", industrialExperience: "0 Years", specialization: "Workshop Technology, Production Engineering" },
        // Computer Science & Engineering
        { name: "Mr. Kunal Mahto", designation: "Lecturer (NBL)", department: "Computer Science & Engineering", deptCode: "cse", qualification: "Ph.D. (Pursuing)", teachingExperience: "9 Years", industrialExperience: "1 Years", specialization: "Data Structures, Algorithms, Emerging Technologies" },
        { name: "Ms. Momita Rani Giri", designation: "Lecturer (NBL)", department: "Computer Science & Engineering", deptCode: "cse", qualification: "B.Tech (CSE)", teachingExperience: "6 Years", industrialExperience: "0 Years", specialization: "Programming, Databases" },
        { name: "Mr. Niraj Kumar", designation: "Lecturer (NBL)", department: "Computer Science & Engineering", deptCode: "cse", qualification: "B.Tech (CSE)", teachingExperience: "4 Years", industrialExperience: "2 Years", specialization: "Networking, Web Technologies" },
        { name: "Ms. Priyanka", designation: "Lecturer (NBL)", department: "Computer Science & Engineering", deptCode: "cse", qualification: "B.Tech (CSE)", teachingExperience: "6 Years", industrialExperience: "0 Years", specialization: "Data Structures, DBMS" },
        // Applied Sciences & Humanities
        { name: "Dr. Chetna Sumedha", designation: "Senior Lecturer", department: "Applied Sciences & Humanities", deptCode: "ash", qualification: "Ph.D. (Pursuing)", teachingExperience: "34 Years", industrialExperience: "0 Years", specialization: "Physics, Applied Sciences" }
    ],

    notices: [
        { title: "4th Semester Attendance Mandatory", date: "2026-04-01", type: "academic", description: "All 4th semester students are hereby directed to ensure regular attendance from 01-04-2026, failing which they will not be permitted to appear in internal and board examinations. Mass bunk will not be allowed under any circumstances.", pdfUrl: "https://gpa.ac.in/cdgps/src/uploads/General%20Postings/Notices/2026/04/950903.pdf" },
        { title: "4th Semester Classes Commence", date: "2026-03-28", type: "academic", description: "All students of 3rd Semester are hereby informed that the classes for 4th Semester will commence from 01-04-2026, and attendance is compulsory for all.", pdfUrl: "https://gpa.ac.in/cdgps/src/uploads/General%20Postings/Notices/2026/03/692166.pdf" },
        { title: "Library Book Tender Notice", date: "2026-04-05", type: "tender", description: "Short-Term Tender Notice for Supply of Diploma-Level Technical, Academic, and General Books for the Library of Government Polytechnic, Adityapur.", pdfUrl: "https://gpa.ac.in/cdgps/src/uploads/General%20Postings/Notices/2026/04/788437.pdf" },
        { title: "JCECEB PECE 2026 — Admissions Open", date: "2026-03-15", type: "admission", description: "Applications for Polytechnic Entrance Competitive Examination (PECE) 2026 are now open. Eligible candidates (10th pass with min 35% marks) must apply online at jceceb.jharkhand.gov.in.", pdfUrl: "" },
        { title: "JUT Board Examination Schedule", date: "2026-03-20", type: "exam", description: "The JUT Board Examinations for 2nd and 4th semester students will be conducted as per the schedule released by Jharkhand University of Technology, Ranchi.", pdfUrl: "" },
        { title: "Anti-Ragging Committee Reconstituted", date: "2026-03-10", type: "general", description: "The Anti-Ragging Committee and Squad for the academic session 2025-26 has been reconstituted. Any ragging incident must be reported immediately.", pdfUrl: "" },
        { title: "Campus Placement Drive — Tata Steel & L&T", date: "2026-02-25", type: "placement", description: "Campus placement drive by Tata Steel and L&T scheduled for March 2026. Eligible final year students must register with the T&P cell.", pdfUrl: "" },
        { title: "E-Kalyan Scholarship Application Deadline", date: "2026-03-01", type: "scholarship", description: "Students belonging to SC/ST/OBC categories are reminded to apply for E-Kalyan Scholarship before the deadline. Apply at ekalyan.cgg.gov.in.", pdfUrl: "" }
    ],

    testimonials: [
        { name: "Amit Kumar", batch: "CSE Graduate", text: "Very Supportive Faculty & Good Placement Opportunities. The teachers really care about student growth and career placement." },
        { name: "Ravi Xaxa", batch: "Mechanical Graduate", text: "The faculty members are well qualified and very supportive. They help students understand concepts clearly and many students get campus placements." },
        { name: "Shiwani Jaiswal", batch: "Electrical Graduate", text: "GPA is one of the best institutes in Jharkhand. Its aim is to achieve excellence in educational field. It provides outstanding infrastructure and facilities." },
        { name: "Sumit Panda", batch: "Metallurgy Graduate", text: "This is one of the best polytechnic colleges in Jharkhand for placements. Major companies like Tata Steel, Wipro, and JSPL visit for recruitment." },
        { name: "Liliy Rugu", batch: "CSE Graduate", text: "The curriculum is relevant to industry needs and makes students industry-ready. Teachers are helpful and the quality of teaching is good." },
        { name: "Imroz Rukhsana", batch: "Electrical Graduate", text: "A life changing experience. The teaching staff provided excellent mentorship and career guidance throughout the program." }
    ]
};


// ─────────────────────────────────────────────────────────────
// PRELOADER
// ─────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
    const preloader = document.getElementById('preloader');
    if (preloader) {
        setTimeout(() => preloader.classList.add('hidden'), 600);
        // Remove from DOM after animation
        setTimeout(() => { if (preloader.parentNode) preloader.parentNode.removeChild(preloader); }, 1400);
    }
});


// ─────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Init AOS
    if (typeof AOS !== 'undefined') {
        AOS.init({
            duration: 650,
            easing: 'ease-out-cubic',
            once: true,
            offset: 40,
            disable: window.innerWidth < 480 ? 'phone' : false
        });
    }

    initNavbar();
    initHeroParticles();
    initCounters();
    renderDepartments();
    renderFaculty();
    renderNotices();
    renderTestimonials();
    initContactForm();
    initBackToTop();
    initSmoothScroll();
});


// ─────────────────────────────────────────────────────────────
// NAVBAR – Sticky + Active section + Mobile menu
// ─────────────────────────────────────────────────────────────
function initNavbar() {
    const navbar = document.getElementById('mainNavbar');
    const toggle = document.getElementById('navToggle');
    const links = document.getElementById('navLinks');
    const navItems = document.querySelectorAll('.nav-link');
    const overlay = document.getElementById('navOverlay');

    if (!navbar || !toggle || !links) return;

    // Scroll: sticky bg + active section
    let ticking = false;
    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(() => {
                const scrollY = window.scrollY;

                // Sticky
                navbar.classList.toggle('scrolled', scrollY > 50);

                // Active section
                let current = 'home';
                document.querySelectorAll('section[id]').forEach(sec => {
                    if (scrollY >= sec.offsetTop - 120) {
                        current = sec.id;
                    }
                });

                navItems.forEach(item => {
                    const href = item.getAttribute('href');
                    item.classList.toggle('active', href === '#' + current);
                });

                ticking = false;
            });
            ticking = true;
        }
    });

    // Mobile toggle
    function closeMobileNav() {
        toggle.classList.remove('active');
        links.classList.remove('open');
        if (overlay) overlay.classList.remove('show');
        document.body.style.overflow = '';
    }

    function openMobileNav() {
        toggle.classList.add('active');
        links.classList.add('open');
        if (overlay) overlay.classList.add('show');
        document.body.style.overflow = 'hidden';
    }

    toggle.addEventListener('click', () => {
        if (links.classList.contains('open')) {
            closeMobileNav();
        } else {
            openMobileNav();
        }
    });

    // Close on overlay click
    if (overlay) {
        overlay.addEventListener('click', closeMobileNav);
    }

    // Close on link click
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            closeMobileNav();
        });
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeMobileNav();
    });
}


// ─────────────────────────────────────────────────────────────
// HERO PARTICLES (lightweight floating dots)
// ─────────────────────────────────────────────────────────────
function initHeroParticles() {
    const container = document.getElementById('heroParticles');
    if (!container) return;

    // Reduce count on mobile for performance
    const count = window.innerWidth < 768 ? 20 : 35;
    const frag = document.createDocumentFragment();

    for (let i = 0; i < count; i++) {
        const p = document.createElement('div');
        p.className = 'hero-particle';
        const size = (Math.random() * 2.5 + 1) + 'px';
        p.style.cssText = `
            left: ${Math.random() * 100}%;
            width: ${size};
            height: ${size};
            animation-duration: ${Math.random() * 18 + 12}s;
            animation-delay: ${Math.random() * 12}s;
        `;
        frag.appendChild(p);
    }

    container.appendChild(frag);
}


// ─────────────────────────────────────────────────────────────
// STAT COUNTERS (Intersection Observer + easeOutCubic)
// ─────────────────────────────────────────────────────────────
function initCounters() {
    const statsSection = document.getElementById('stats');
    if (!statsSection) return;

    const counters = statsSection.querySelectorAll('.stat-number');
    let animated = false;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && !animated) {
                animated = true;
                counters.forEach(el => animateCounter(el));
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.25 });

    observer.observe(statsSection);
}

function animateCounter(el) {
    const target = parseInt(el.dataset.target, 10);
    const suffix = el.dataset.suffix || '';
    const duration = 2200;
    const start = performance.now();

    // Don't use locale for "established" year
    const isYear = target >= 1900 && target <= 2100;

    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(eased * target);

        el.textContent = (isYear ? current : current.toLocaleString('en-IN')) + suffix;

        if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
}


// ─────────────────────────────────────────────────────────────
// DEPARTMENTS
// ─────────────────────────────────────────────────────────────
function renderDepartments() {
    const container = document.getElementById('departmentsContainer');
    if (!container) return;

    const icons = { cse: 'fa-microchip', mech: 'fa-gears', ee: 'fa-bolt', met: 'fa-hammer' };

    container.innerHTML = DATA.departments.map((d, i) => `
        <div class="dept-card" data-aos="fade-up" data-aos-delay="${i * 80}">
            <div class="dept-card-header">
                <div class="dept-icon ${d.code}">
                    <i class="fas ${icons[d.code] || 'fa-building-columns'}"></i>
                </div>
                <div>
                    <h3 class="dept-card-title">${d.name}</h3>
                    <span class="dept-card-code">${d.code.toUpperCase()} · Diploma</span>
                </div>
            </div>
            <div class="dept-card-body">
                <p>${d.description}</p>
                <div class="dept-meta">
                    <span class="dept-meta-item"><i class="fas fa-users"></i> ${d.seats} Seats</span>
                    <span class="dept-meta-item"><i class="fas fa-clock"></i> ${d.duration}</span>
                    <span class="dept-meta-item"><i class="fas fa-graduation-cap"></i> 10th Pass</span>
                </div>
                <div class="dept-subjects">
                    ${d.subjects.map(s => `<span class="dept-subject-tag">${s}</span>`).join('')}
                </div>
            </div>
        </div>
    `).join('');
}


// ─────────────────────────────────────────────────────────────
// FACULTY (with staggered animation on filter change)
// ─────────────────────────────────────────────────────────────
let facultyFilterInitialized = false;

function renderFaculty(filterCode) {
    filterCode = filterCode || 'All';

    const container = document.getElementById('facultyContainer');
    const filter = document.getElementById('facultyFilter');
    if (!container || !filter) return;

    // Populate filter dropdown (once)
    if (!facultyFilterInitialized) {
        const deptCodes = [...new Set(DATA.faculty.map(f => f.deptCode))];
        const deptNames = {
            ee: 'Electrical Engineering',
            mech: 'Mechanical Engineering',
            met: 'Metallurgical Engineering',
            cse: 'Computer Science & Engineering',
            ash: 'Applied Sciences & Humanities'
        };

        filter.innerHTML = '<option value="All">All Departments</option>' +
            deptCodes.map(code =>
                `<option value="${code}">${deptNames[code] || code.toUpperCase()}</option>`
            ).join('');

        filter.addEventListener('change', (e) => renderFaculty(e.target.value));
        facultyFilterInitialized = true;
    }

    const list = filterCode === 'All'
        ? DATA.faculty
        : DATA.faculty.filter(f => f.deptCode === filterCode);

    // Build HTML with staggered animation delay
    container.innerHTML = list.map((f, i) => {
        const initials = f.name.replace(/^(Mr\.|Ms\.|Dr\.|Md\.)\s*/i, '')
            .split(' ')
            .filter(w => w.length > 0)
            .map(w => w[0])
            .join('')
            .substring(0, 2)
            .toUpperCase();

        return `
        <div class="faculty-card" style="animation-delay: ${i * 50}ms">
            <div class="faculty-card-header">
                <div class="faculty-avatar">${initials}</div>
                <div>
                    <h4 class="faculty-name">${f.name}</h4>
                    <span class="faculty-designation">${f.designation}</span>
                </div>
            </div>
            <div class="faculty-detail"><strong>Department:</strong><span>${f.department}</span></div>
            <div class="faculty-detail"><strong>Qualification:</strong><span>${f.qualification}</span></div>
            <div class="faculty-detail"><strong>Teaching:</strong><span>${f.teachingExperience}</span></div>
            <div class="faculty-detail"><strong>Industrial:</strong><span>${f.industrialExperience}</span></div>
            <div class="faculty-detail"><strong>Specialize:</strong><span>${f.specialization}</span></div>
        </div>
        `;
    }).join('');
}


// ─────────────────────────────────────────────────────────────
// NOTICES
// ─────────────────────────────────────────────────────────────
function renderNotices() {
    const container = document.getElementById('noticesList');
    if (!container) return;

    const sorted = [...DATA.notices].sort((a, b) => new Date(b.date) - new Date(a.date));
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];

    container.innerHTML = sorted.map((n, i) => {
        const d = new Date(n.date);
        const day = d.getDate();
        const monthYear = months[d.getMonth()] + ' ' + d.getFullYear();

        const pdfBtn = n.pdfUrl
            ? `<a href="${n.pdfUrl}" target="_blank" rel="noopener" class="notice-pdf-btn"><i class="fas fa-file-pdf"></i> View PDF</a>`
            : '';

        return `
        <div class="notice-card" data-aos="fade-up" data-aos-delay="${i * 50}">
            <div class="notice-date-badge">
                <span class="day">${day}</span>
                <span class="month-year">${monthYear}</span>
            </div>
            <div class="notice-content">
                <h4>${n.title}</h4>
                <p>${n.description}</p>
                <div class="notice-footer">
                    <span class="notice-type ${n.type}">${n.type}</span>
                    ${pdfBtn}
                </div>
            </div>
        </div>
        `;
    }).join('');
}


// ─────────────────────────────────────────────────────────────
// TESTIMONIALS CAROUSEL (smooth, touch-friendly)
// ─────────────────────────────────────────────────────────────
function renderTestimonials() {
    const track = document.getElementById('carouselTrack');
    const dotsContainer = document.getElementById('carouselDots');
    const prevBtn = document.getElementById('prevTestimonial');
    const nextBtn = document.getElementById('nextTestimonial');
    if (!track || !dotsContainer || !prevBtn || !nextBtn) return;

    let currentSlide = 0;
    const total = DATA.testimonials.length;

    // Render slides
    track.innerHTML = DATA.testimonials.map(t => `
        <div class="testimonial-slide">
            <p class="testimonial-quote">${t.text}</p>
            <h5 class="testimonial-author">${t.name}</h5>
            <span class="testimonial-batch">${t.batch}</span>
        </div>
    `).join('');

    // Render dots
    dotsContainer.innerHTML = DATA.testimonials.map((_, i) =>
        `<button class="carousel-dot ${i === 0 ? 'active' : ''}" data-slide="${i}" aria-label="Slide ${i + 1}"></button>`
    ).join('');

    const dots = dotsContainer.querySelectorAll('.carousel-dot');

    function goToSlide(index) {
        currentSlide = ((index % total) + total) % total;
        track.style.transform = `translateX(-${currentSlide * 100}%)`;
        dots.forEach((dot, i) => dot.classList.toggle('active', i === currentSlide));
    }

    prevBtn.addEventListener('click', () => goToSlide(currentSlide - 1));
    nextBtn.addEventListener('click', () => goToSlide(currentSlide + 1));

    dots.forEach(dot => {
        dot.addEventListener('click', () => goToSlide(parseInt(dot.dataset.slide, 10)));
    });

    // Auto-play with pause on hover/focus
    let autoTimer = setInterval(() => goToSlide(currentSlide + 1), 5000);

    const carousel = document.getElementById('testimonialsCarousel');
    if (carousel) {
        const pause = () => clearInterval(autoTimer);
        const resume = () => { autoTimer = setInterval(() => goToSlide(currentSlide + 1), 5000); };

        carousel.addEventListener('mouseenter', pause);
        carousel.addEventListener('mouseleave', resume);
        carousel.addEventListener('focusin', pause);
        carousel.addEventListener('focusout', resume);
    }

    // Touch/swipe support
    let touchStartX = 0;
    let touchEndX = 0;

    track.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });

    track.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        const diff = touchStartX - touchEndX;
        if (Math.abs(diff) > 50) {
            if (diff > 0) goToSlide(currentSlide + 1);
            else goToSlide(currentSlide - 1);
        }
    }, { passive: true });

    // Keyboard navigation (when carousel is focused)
    if (carousel) {
        carousel.setAttribute('tabindex', '0');
        carousel.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') goToSlide(currentSlide - 1);
            if (e.key === 'ArrowRight') goToSlide(currentSlide + 1);
        });
    }
}


// ─────────────────────────────────────────────────────────────
// CONTACT FORM
// ─────────────────────────────────────────────────────────────
function initContactForm() {
    const form = document.getElementById('contactForm');
    const alertBox = document.getElementById('contactAlert');
    if (!form || !alertBox) return;

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const name = form.elements['name'].value.trim();
        const email = form.elements['email'].value.trim();
        const message = form.elements['message'].value.trim();

        if (!name || !email || !message) {
            showAlert('All fields are required.', 'danger');
            return;
        }

        // Basic email validation
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
            showAlert('Please enter a valid email address.', 'danger');
            return;
        }

        // Store (same as Java server)
        try {
            const stored = JSON.parse(localStorage.getItem('contactMessages') || '[]');
            stored.push({ name, email, message, timestamp: new Date().toISOString() });
            localStorage.setItem('contactMessages', JSON.stringify(stored));
        } catch (err) {
            // localStorage might be full or unavailable — gracefully ignore
        }

        showAlert('Thank you! Your message has been received. We will get back to you soon.', 'success');
        form.reset();

        // Focus back to first field
        form.elements['name'].focus();
    });

    function showAlert(msg, type) {
        alertBox.innerHTML = `<div class="alert alert-${type}">${msg}</div>`;
        setTimeout(() => {
            const alert = alertBox.querySelector('.alert');
            if (alert) {
                alert.style.opacity = '0';
                alert.style.transform = 'translateY(-8px)';
                alert.style.transition = 'all 0.3s ease';
                setTimeout(() => { alertBox.innerHTML = ''; }, 300);
            }
        }, 4500);
    }
}


// ─────────────────────────────────────────────────────────────
// BACK TO TOP
// ─────────────────────────────────────────────────────────────
function initBackToTop() {
    const btn = document.getElementById('backToTop');
    if (!btn) return;

    let ticking = false;
    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(() => {
                btn.classList.toggle('visible', window.scrollY > 350);
                ticking = false;
            });
            ticking = true;
        }
    });

    btn.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
}


// ─────────────────────────────────────────────────────────────
// SMOOTH SCROLL (for all anchor links)
// ─────────────────────────────────────────────────────────────
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', (e) => {
            const href = anchor.getAttribute('href');
            if (href === '#') return;

            const target = document.querySelector(href);
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}
