"""
Population-Aligned Persona Generation for LLM-based Social Simulation.

Based on:
- "Population-Aligned Persona Generation for LLM-based Social Simulation" (Microsoft Research)
- "Polypersona: Persona-Grounded LLM for Synthetic Survey Responses"

Key improvements over simple demographic sampling:
1. Generate narrative personas from demographic + psychographic data
2. Quality assessment for each persona
3. Importance sampling to match Big Five personality distribution
4. Address WEIRD bias through diverse demographic coverage
"""

import json
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PersonaTrait:
    """Big Five personality trait scores."""
    openness: float  # 0-1
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'PersonaTrait':
        return cls(
            openness=d.get('openness', 0.5),
            conscientiousness=d.get('conscientiousness', 0.5),
            extraversion=d.get('extraversion', 0.5),
            agreeableness=d.get('agreeableness', 0.5),
            neuroticism=d.get('neuroticism', 0.5),
        )


@dataclass
class Persona:
    """A complete persona with demographics, traits, and narrative."""
    id: str
    name: str
    age: int
    gender: str
    occupation: str
    income_bracket: str
    education: str
    location: str
    traits: PersonaTrait
    narrative: str
    economic_preferences: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0

    def to_prompt(self) -> str:
        """Convert persona to a prompt string for LLM."""
        trait_desc = []
        if self.traits.openness > 0.6:
            trait_desc.append("open to new experiences")
        if self.traits.conscientiousness > 0.6:
            trait_desc.append("highly organized and disciplined")
        if self.traits.extraversion > 0.6:
            trait_desc.append("outgoing and sociable")
        if self.traits.agreeableness > 0.6:
            trait_desc.append("cooperative and trusting")
        if self.traits.neuroticism > 0.6:
            trait_desc.append("emotionally sensitive")

        traits_str = ", ".join(trait_desc) if trait_desc else "balanced personality"

        return f"""You are {self.name}, a {self.age}-year-old {self.gender} working as a {self.occupation}.

Background: {self.narrative}

Your personality: {traits_str}

Income bracket: {self.income_bracket}
Education: {self.education}
Location: {self.location}

Your economic preferences:
- Risk tolerance: {self.economic_preferences.get('risk_tolerance', 'moderate')}
- Tax attitude: {self.economic_preferences.get('tax_attitude', 'neutral')}
- Work-life balance priority: {self.economic_preferences.get('work_life_balance', 'moderate')}
"""

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'occupation': self.occupation,
            'income_bracket': self.income_bracket,
            'education': self.education,
            'location': self.location,
            'traits': self.traits.to_dict(),
            'narrative': self.narrative,
            'economic_preferences': self.economic_preferences,
            'quality_score': self.quality_score,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Persona':
        return cls(
            id=d['id'],
            name=d['name'],
            age=d['age'],
            gender=d['gender'],
            occupation=d['occupation'],
            income_bracket=d['income_bracket'],
            education=d['education'],
            location=d['location'],
            traits=PersonaTrait.from_dict(d['traits']),
            narrative=d['narrative'],
            economic_preferences=d.get('economic_preferences', {}),
            quality_score=d.get('quality_score', 1.0),
        )


# Target Big Five distribution based on population studies
# (Mean, Std) for each trait on 0-1 scale
TARGET_BIG_FIVE_DISTRIBUTION = {
    'openness': (0.58, 0.18),
    'conscientiousness': (0.55, 0.17),
    'extraversion': (0.52, 0.19),
    'agreeableness': (0.60, 0.16),
    'neuroticism': (0.45, 0.20),
}

# Income brackets aligned with US tax system
INCOME_BRACKETS = [
    ('$0-$23,000', 0.20),      # ~20% of population
    ('$23,000-$47,000', 0.25),  # ~25%
    ('$47,000-$94,000', 0.30),  # ~30%
    ('$94,000-$192,000', 0.18), # ~18%
    ('$192,000-$500,000', 0.05),# ~5%
    ('$500,000+', 0.02),        # ~2%
]

# Occupation categories with typical traits and tax attitudes
OCCUPATION_PROFILES = {
    'software_engineer': {
        'trait_bias': {'openness': 0.1, 'conscientiousness': 0.1},
        'tax_attitude': 'moderate',
        'income_range': (3, 5),  # Index into INCOME_BRACKETS
    },
    'teacher': {
        'trait_bias': {'agreeableness': 0.15, 'conscientiousness': 0.1},
        'tax_attitude': 'supportive',
        'income_range': (1, 3),
    },
    'entrepreneur': {
        'trait_bias': {'openness': 0.2, 'extraversion': 0.15},
        'tax_attitude': 'resistant',
        'income_range': (3, 6),
    },
    'healthcare_worker': {
        'trait_bias': {'agreeableness': 0.2, 'conscientiousness': 0.15},
        'tax_attitude': 'supportive',
        'income_range': (2, 5),
    },
    'retail_worker': {
        'trait_bias': {'extraversion': 0.1},
        'tax_attitude': 'neutral',
        'income_range': (0, 2),
    },
    'financial_analyst': {
        'trait_bias': {'conscientiousness': 0.15},
        'tax_attitude': 'resistant',
        'income_range': (3, 5),
    },
    'artist': {
        'trait_bias': {'openness': 0.25, 'neuroticism': 0.1},
        'tax_attitude': 'supportive',
        'income_range': (0, 3),
    },
    'construction_worker': {
        'trait_bias': {'conscientiousness': 0.1},
        'tax_attitude': 'moderate',
        'income_range': (1, 3),
    },
    'lawyer': {
        'trait_bias': {'conscientiousness': 0.15, 'extraversion': 0.1},
        'tax_attitude': 'resistant',
        'income_range': (4, 6),
    },
    'government_employee': {
        'trait_bias': {'conscientiousness': 0.1, 'agreeableness': 0.1},
        'tax_attitude': 'supportive',
        'income_range': (2, 4),
    },
}

# Names by gender (simplified)
NAMES = {
    'male': ['James', 'John', 'Robert', 'Michael', 'David', 'William', 'Richard', 'Joseph',
             'Thomas', 'Christopher', 'Daniel', 'Matthew', 'Anthony', 'Mark', 'Steven',
             'Andrew', 'Kenneth', 'Joshua', 'Kevin', 'Brian', 'Carlos', 'Jose', 'Wei',
             'Ahmed', 'Raj', 'Dmitri', 'Takeshi', 'Emmanuel', 'Omar', 'Yusuf'],
    'female': ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan',
               'Jessica', 'Sarah', 'Karen', 'Lisa', 'Nancy', 'Betty', 'Margaret', 'Sandra',
               'Ashley', 'Dorothy', 'Kimberly', 'Emily', 'Donna', 'Maria', 'Mei', 'Fatima',
               'Priya', 'Olga', 'Yuki', 'Amara', 'Leila', 'Sofia', 'Aaliyah'],
}


class PopulationAlignedPersonaGenerator:
    """
    Generate personas aligned with population distributions.

    Key features:
    1. Sample demographics from realistic distributions
    2. Generate Big Five traits with occupation-based biases
    3. Importance sampling to match target personality distribution
    4. Quality filtering for coherent personas
    """

    def __init__(
        self,
        llm_model: Optional[Any] = None,
        target_distribution: Dict = None,
        quality_threshold: float = 0.7,
        use_llm_narratives: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the persona generator.

        Args:
            llm_model: LLM model for generating narratives (optional)
            target_distribution: Target Big Five distribution (defaults to population norms)
            quality_threshold: Minimum quality score for personas
            use_llm_narratives: Whether to use LLM for narrative generation
            seed: Random seed for reproducibility
        """
        self.llm_model = llm_model
        self.target_distribution = target_distribution or TARGET_BIG_FIVE_DISTRIBUTION
        self.quality_threshold = quality_threshold
        self.use_llm_narratives = use_llm_narratives
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self._persona_counter = 0

    def generate_personas(
        self,
        n: int,
        occupation_distribution: Optional[Dict[str, float]] = None,
        income_distribution: Optional[List[Tuple[str, float]]] = None,
    ) -> List[Persona]:
        """
        Generate n population-aligned personas.

        Args:
            n: Number of personas to generate
            occupation_distribution: Custom occupation weights
            income_distribution: Custom income bracket weights

        Returns:
            List of Persona objects
        """
        logger.info(f"Generating {n} population-aligned personas")

        # Use default distributions if not provided
        if occupation_distribution is None:
            occupation_distribution = {k: 1.0 / len(OCCUPATION_PROFILES)
                                      for k in OCCUPATION_PROFILES}
        if income_distribution is None:
            income_distribution = INCOME_BRACKETS

        # Step 1: Generate raw personas
        raw_personas = []
        for _ in range(int(n * 1.5)):  # Generate extra for filtering
            persona = self._generate_single_persona(
                occupation_distribution,
                income_distribution
            )
            raw_personas.append(persona)

        # Step 2: Quality filtering
        quality_personas = [p for p in raw_personas if p.quality_score >= self.quality_threshold]
        logger.info(f"Quality filter: {len(quality_personas)}/{len(raw_personas)} passed")

        # Step 3: Importance sampling for population alignment
        aligned_personas = self._importance_sample(quality_personas, n)
        logger.info(f"Generated {len(aligned_personas)} aligned personas")

        return aligned_personas

    def _generate_single_persona(
        self,
        occupation_dist: Dict[str, float],
        income_dist: List[Tuple[str, float]],
    ) -> Persona:
        """Generate a single persona."""
        self._persona_counter += 1

        # Sample occupation
        occupations = list(occupation_dist.keys())
        occupation_weights = [occupation_dist[o] for o in occupations]
        occupation = self.rng.choice(occupations, p=np.array(occupation_weights) / sum(occupation_weights))
        occ_profile = OCCUPATION_PROFILES[occupation]

        # Sample income bracket (constrained by occupation)
        min_bracket, max_bracket = occ_profile['income_range']
        valid_brackets = income_dist[min_bracket:max_bracket + 1]
        if valid_brackets:
            bracket_names, bracket_weights = zip(*valid_brackets)
            income_bracket = self.rng.choice(
                bracket_names,
                p=np.array(bracket_weights) / sum(bracket_weights)
            )
        else:
            income_bracket = income_dist[0][0]

        # Sample demographics
        gender = self.rng.choice(['male', 'female'])
        name = random.choice(NAMES[gender])
        age = int(self.rng.integers(22, 65))

        education_options = ['High School', 'Some College', 'Bachelor\'s', 'Master\'s', 'PhD']
        education_weights = [0.25, 0.20, 0.35, 0.15, 0.05]
        education = self.rng.choice(education_options, p=education_weights)

        location_options = ['Urban', 'Suburban', 'Rural']
        location_weights = [0.35, 0.45, 0.20]
        location = self.rng.choice(location_options, p=location_weights)

        # Generate Big Five traits with occupation bias
        traits = self._generate_traits(occ_profile)

        # Generate economic preferences
        economic_preferences = self._generate_economic_preferences(
            occupation, income_bracket, traits
        )

        # Generate narrative
        narrative = self._generate_narrative(
            name, age, gender, occupation, income_bracket, education, traits
        )

        # Calculate quality score
        quality_score = self._calculate_quality_score(
            traits, occupation, income_bracket, narrative
        )

        return Persona(
            id=f"persona_{self._persona_counter}",
            name=name,
            age=age,
            gender=gender,
            occupation=occupation.replace('_', ' ').title(),
            income_bracket=income_bracket,
            education=education,
            location=location,
            traits=traits,
            narrative=narrative,
            economic_preferences=economic_preferences,
            quality_score=quality_score,
        )

    def _generate_traits(self, occupation_profile: Dict) -> PersonaTrait:
        """Generate Big Five traits with occupation-based biases."""
        trait_bias = occupation_profile.get('trait_bias', {})

        traits = {}
        for trait, (mean, std) in self.target_distribution.items():
            # Apply occupation bias
            bias = trait_bias.get(trait, 0.0)
            value = self.rng.normal(mean + bias, std)
            # Clip to [0, 1]
            traits[trait] = float(np.clip(value, 0.0, 1.0))

        return PersonaTrait(**traits)

    def _generate_economic_preferences(
        self,
        occupation: str,
        income_bracket: str,
        traits: PersonaTrait,
    ) -> Dict[str, str]:
        """Generate economic preferences based on occupation and traits."""
        occ_profile = OCCUPATION_PROFILES.get(occupation, {})

        # Risk tolerance based on openness and income
        high_income = any(x in income_bracket for x in ['192,000', '500,000'])
        if traits.openness > 0.6 and high_income:
            risk_tolerance = 'high'
        elif traits.openness < 0.4 or traits.neuroticism > 0.6:
            risk_tolerance = 'low'
        else:
            risk_tolerance = 'moderate'

        # Tax attitude from occupation profile
        tax_attitude = occ_profile.get('tax_attitude', 'neutral')
        # Modify based on agreeableness
        if traits.agreeableness > 0.7 and tax_attitude == 'resistant':
            tax_attitude = 'moderate'
        elif traits.agreeableness < 0.3 and tax_attitude == 'supportive':
            tax_attitude = 'moderate'

        # Work-life balance based on conscientiousness and extraversion
        if traits.conscientiousness > 0.7:
            work_life = 'work-focused'
        elif traits.extraversion > 0.6:
            work_life = 'life-focused'
        else:
            work_life = 'moderate'

        return {
            'risk_tolerance': risk_tolerance,
            'tax_attitude': tax_attitude,
            'work_life_balance': work_life,
        }

    def _generate_narrative(
        self,
        name: str,
        age: int,
        gender: str,
        occupation: str,
        income_bracket: str,
        education: str,
        traits: PersonaTrait,
    ) -> str:
        """Generate a narrative description of the persona."""
        if self.use_llm_narratives and self.llm_model is not None:
            return self._llm_generate_narrative(
                name, age, gender, occupation, income_bracket, education, traits
            )

        # Template-based narrative
        occupation_display = occupation.replace('_', ' ')

        trait_descriptors = []
        if traits.openness > 0.6:
            trait_descriptors.append("curious and creative")
        if traits.conscientiousness > 0.6:
            trait_descriptors.append("organized and reliable")
        if traits.extraversion > 0.6:
            trait_descriptors.append("outgoing")
        if traits.agreeableness > 0.6:
            trait_descriptors.append("empathetic")

        personality = ", ".join(trait_descriptors) if trait_descriptors else "balanced"

        career_stage = "early-career" if age < 30 else "mid-career" if age < 50 else "experienced"

        narrative = f"{name} is a {career_stage} {occupation_display} with {education} education. "
        narrative += f"They are {personality} and earn in the {income_bracket} range. "

        # Add economic context
        if "500,000" in income_bracket or "192,000" in income_bracket:
            narrative += "As a high earner, they are keenly aware of tax policy impacts on their income. "
        elif "$0-$23,000" in income_bracket:
            narrative += "They rely on government programs and tax credits to make ends meet. "

        return narrative

    def _llm_generate_narrative(
        self,
        name: str,
        age: int,
        gender: str,
        occupation: str,
        income_bracket: str,
        education: str,
        traits: PersonaTrait,
    ) -> str:
        """Use LLM to generate a richer narrative."""
        prompt = f"""Generate a brief (2-3 sentence) narrative background for this person:
Name: {name}
Age: {age}
Gender: {gender}
Occupation: {occupation.replace('_', ' ')}
Income: {income_bracket}
Education: {education}

Personality traits (0-1 scale):
- Openness: {traits.openness:.2f}
- Conscientiousness: {traits.conscientiousness:.2f}
- Extraversion: {traits.extraversion:.2f}
- Agreeableness: {traits.agreeableness:.2f}
- Neuroticism: {traits.neuroticism:.2f}

Focus on their work history, life situation, and attitude toward work and taxes.
Be specific and realistic. No generic statements."""

        try:
            response, _ = self.llm_model.send_msg(
                "You are a demographic researcher creating realistic personas.",
                prompt,
                temperature=0.8
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM narrative generation failed: {e}")
            return self._generate_narrative(
                name, age, gender, occupation, income_bracket, education, traits
            )

    def _calculate_quality_score(
        self,
        traits: PersonaTrait,
        occupation: str,
        income_bracket: str,
        narrative: str,
    ) -> float:
        """
        Calculate quality score for a persona.

        Checks for:
        - Trait-occupation coherence
        - Income-occupation coherence
        - Narrative quality
        """
        score = 1.0

        # Check trait-occupation coherence
        occ_profile = OCCUPATION_PROFILES.get(occupation.lower().replace(' ', '_'), {})
        trait_bias = occ_profile.get('trait_bias', {})

        for trait_name, expected_bias in trait_bias.items():
            actual_value = getattr(traits, trait_name)
            # Penalize if trait is opposite of expected
            if expected_bias > 0 and actual_value < 0.3:
                score -= 0.1
            elif expected_bias < 0 and actual_value > 0.7:
                score -= 0.1

        # Check income-occupation coherence
        income_range = occ_profile.get('income_range', (0, 5))
        bracket_idx = next(
            (i for i, (b, _) in enumerate(INCOME_BRACKETS) if b == income_bracket),
            2
        )
        if bracket_idx < income_range[0] - 1 or bracket_idx > income_range[1] + 1:
            score -= 0.2

        # Narrative quality (basic checks)
        if len(narrative) < 50:
            score -= 0.1
        if len(narrative) > 500:
            score -= 0.1

        return max(0.0, min(1.0, score))

    def _importance_sample(
        self,
        personas: List[Persona],
        n: int,
    ) -> List[Persona]:
        """
        Use importance sampling to align persona distribution with target.

        This ensures the final persona set matches population-level
        Big Five distributions.
        """
        if len(personas) <= n:
            return personas

        # Calculate importance weights based on Big Five distribution
        weights = []
        for persona in personas:
            weight = 1.0
            for trait_name, (target_mean, target_std) in self.target_distribution.items():
                actual_value = getattr(persona.traits, trait_name)
                # Calculate how likely this trait value is under target distribution
                z_score = (actual_value - target_mean) / target_std
                likelihood = np.exp(-0.5 * z_score ** 2)
                weight *= likelihood
            weights.append(weight * persona.quality_score)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Sample without replacement
        indices = self.rng.choice(
            len(personas),
            size=min(n, len(personas)),
            replace=False,
            p=weights
        )

        return [personas[i] for i in indices]

    def save_personas(self, personas: List[Persona], filepath: str):
        """Save personas to JSON file."""
        data = [p.to_dict() for p in personas]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(personas)} personas to {filepath}")

    def load_personas(self, filepath: str) -> List[Persona]:
        """Load personas from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        personas = [Persona.from_dict(d) for d in data]
        logger.info(f"Loaded {len(personas)} personas from {filepath}")
        return personas


def generate_aligned_personas(
    n: int,
    llm_model: Optional[Any] = None,
    seed: int = 42,
    use_llm_narratives: bool = False,
) -> Dict[str, str]:
    """
    Convenience function to generate personas in the format expected by worker.py.

    Returns:
        Dict mapping persona_id to persona prompt string
    """
    generator = PopulationAlignedPersonaGenerator(
        llm_model=llm_model,
        use_llm_narratives=use_llm_narratives,
        seed=seed,
    )

    personas = generator.generate_personas(n)

    return {p.id: p.to_prompt() for p in personas}
