import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Brain, BarChart3, FileText, Rocket, Sparkles, Zap, Target } from 'lucide-react';

interface HeroPageProps {
  onGetStarted: () => void;
}

const HeroPage: React.FC<HeroPageProps> = ({ onGetStarted }) => {
  const [cursorPos, setCursorPos] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setCursorPos({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Analysis",
      description: "Advanced machine learning algorithms analyze your data with unprecedented accuracy and speed"
    },
    {
      icon: BarChart3,
      title: "Rich Visualizations",
      description: "Interactive charts and graphs that bring your data insights to life with stunning clarity"
    },
    {
      icon: FileText,
      title: "Instant Reports",
      description: "Generate comprehensive reports in seconds with actionable insights and recommendations"
    }
  ];

  const stats = [
    { value: "99.9%", label: "Accuracy Rate", icon: Target },
    { value: "50ms", label: "Average Response", icon: Zap },
    { value: "24/7", label: "AI Processing", icon: Rocket }
  ];

  return (
    <div className="min-h-screen bg-gradient-cosmic relative overflow-hidden">
      {/* Glass Particles Background */}
      <div className="glass-particles" />
      
      {/* Dynamic Cursor Effect */}
      {isHovering && (
        <motion.div
          className="fixed pointer-events-none z-50"
          style={{
            left: cursorPos.x - 20,
            top: cursorPos.y - 20,
          }}
          animate={{
            scale: [1, 1.5, 1],
            opacity: [0.5, 1, 0.5],
          }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
          }}
        >
          <div className="w-10 h-10 rounded-full bg-gradient-neon opacity-30 blur-sm" />
        </motion.div>
      )}

      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(30)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-primary rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`
            }}
            animate={{
              scale: [0, 1.5, 0],
              opacity: [0, 1, 0],
              rotate: [0, 180, 360],
            }}
            transition={{
              duration: 3,
              repeat: Infinity,
              repeatDelay: Math.random() * 3,
              delay: Math.random() * 2
            }}
            initial={{ scale: 0, opacity: 0 }}
          />
        ))}
      </div>

      <motion.div
        className="relative z-10 container mx-auto px-4 py-16 lg:py-24"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1.2 }}
      >
        {/* Hero Section */}
        <div className="text-center mb-20">
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="relative inline-block mb-8"
          >
            <motion.div
              animate={{ rotate: [0, 360] }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="absolute -top-6 -right-6"
            >
              <Sparkles className="w-10 h-10 text-accent animate-glow-pulse" />
            </motion.div>
            <h1 className="text-6xl lg:text-8xl font-bold bg-gradient-neon bg-clip-text text-transparent animate-gradient-shift bg-[length:200%_200%] tracking-tight">
              Smart Dataset
              <br />
              <span className="text-5xl lg:text-7xl">Analyzer</span>
            </h1>
          </motion.div>
          
          <motion.p 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.5 }}
            className="text-xl lg:text-3xl text-muted-foreground mb-12 max-w-4xl mx-auto leading-relaxed"
          >
            Transform your raw data into{" "}
            <span className="text-primary font-semibold animate-glow-pulse">powerful insights</span>{" "}
            with our AI-driven analysis platform. Unlock patterns, trends, and intelligence hidden in your datasets.
          </motion.p>

          <motion.div 
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            <Button
              variant="hero"
              size="xl"
              onClick={onGetStarted}
              className="group relative overflow-hidden"
              onMouseEnter={() => setIsHovering(true)}
              onMouseLeave={() => setIsHovering(false)}
            >
              <span className="flex items-center gap-4 text-xl font-bold">
                Try Now 
                <motion.div
                  animate={{ 
                    rotate: [0, 15, -15, 0],
                    scale: [1, 1.1, 1]
                  }}
                  transition={{ 
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                >
                  <Rocket className="w-6 h-6" />
                </motion.div>
              </span>
              <motion.div 
                className="absolute inset-0 bg-gradient-neon opacity-0 group-hover:opacity-30"
                animate={{ x: ["-100%", "100%"] }}
                transition={{ 
                  duration: 2,
                  repeat: Infinity,
                  ease: "linear"
                }}
              />
            </Button>
          </motion.div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 max-w-7xl mx-auto mb-20">
          {features.map((feature, index) => (
            <motion.div 
              key={index}
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 1.0 + index * 0.2 }}
            >
              <Card className="glass-card hover-lift group h-full">
                <CardContent className="p-10 text-center">
                  <motion.div 
                    className="mb-8 flex justify-center"
                    whileHover={{ rotate: [0, -10, 10, 0] }}
                    transition={{ duration: 0.5 }}
                  >
                    <div className="w-20 h-20 bg-gradient-hero rounded-full flex items-center justify-center group-hover:shadow-glow-primary transition-all duration-500 group-hover:scale-110">
                      <feature.icon className="w-10 h-10 text-primary group-hover:animate-glow-pulse" />
                    </div>
                  </motion.div>
                  <h3 className="text-2xl font-bold mb-6 text-foreground group-hover:text-primary transition-colors">
                    {feature.title}
                  </h3>
                  <p className="text-muted-foreground leading-relaxed text-lg">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Enhanced Stats Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-4xl mx-auto">
          {stats.map((stat, index) => (
            <motion.div 
              key={index}
              initial={{ opacity: 0, scale: 0.5 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ 
                duration: 0.6, 
                delay: 1.6 + index * 0.1,
                type: "spring",
                stiffness: 200
              }}
              className="text-center group glass-card p-6 hover-lift"
            >
              <motion.div
                className="mb-4 flex justify-center"
                whileHover={{ scale: 1.2, rotate: 360 }}
                transition={{ duration: 0.5 }}
              >
                <stat.icon className="w-8 h-8 text-accent" />
              </motion.div>
              <motion.div 
                className="text-4xl lg:text-5xl font-bold text-primary mb-3 group-hover:animate-glow-pulse"
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 1.8 + index * 0.1, type: "spring" }}
              >
                {stat.value}
              </motion.div>
              <div className="text-muted-foreground text-base lg:text-lg font-medium">
                {stat.label}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 2.2 }}
          className="text-center mt-20"
        >
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Join thousands of data scientists and analysts who trust our platform for their most critical insights.
          </p>
          <motion.div
            animate={{ y: [0, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          >
            <div className="text-primary text-lg font-semibold">
              ↓ Start Your Analysis Journey ↓
            </div>
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default HeroPage;