use seize::Collector;
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use std::borrow::Borrow;
use std::fmt::{self, Formatter};
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::{Guard, HashMap, HashMapRef, HashSet, HashSetRef};

struct MapVisitor<K, V, S, C = Collector>
where
    C: Borrow<Collector>,
{
    _marker: PhantomData<HashMap<K, V, S, C>>,
}

impl<K, V, S, C, G> Serialize for HashMapRef<'_, K, V, S, C, G>
where
    K: Serialize + Hash + Eq,
    V: Serialize,
    G: Guard,
    C: Borrow<Collector>,
    S: BuildHasher,
{
    fn serialize<Sr>(&self, serializer: Sr) -> Result<Sr::Ok, Sr::Error>
    where
        Sr: Serializer,
    {
        serializer.collect_map(self)
    }
}

impl<K, V, S, C> Serialize for HashMap<K, V, S, C>
where
    K: Serialize + Hash + Eq,
    V: Serialize,
    S: BuildHasher,
    C: Borrow<Collector>,
{
    fn serialize<Sr>(&self, serializer: Sr) -> Result<Sr::Ok, Sr::Error>
    where
        Sr: Serializer,
    {
        self.pin().serialize(serializer)
    }
}

impl<'de, K, V, S> Deserialize<'de> for HashMap<K, V, S>
where
    K: Deserialize<'de> + Hash + Eq,
    V: Deserialize<'de>,
    S: Default + BuildHasher,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_map(MapVisitor::new())
    }
}

impl<K, V, S> MapVisitor<K, V, S> {
    pub(crate) fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<'de, K, V, S> Visitor<'de> for MapVisitor<K, V, S>
where
    K: Deserialize<'de> + Hash + Eq,
    V: Deserialize<'de>,
    S: Default + BuildHasher,
{
    type Value = HashMap<K, V, S>;

    fn expecting(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "a map")
    }

    fn visit_map<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: MapAccess<'de>,
    {
        let values = match access.size_hint() {
            Some(size) => HashMap::with_capacity_and_hasher(size, S::default()),
            None => HashMap::default(),
        };

        {
            let values = values.pin();
            while let Some((key, value)) = access.next_entry()? {
                values.insert(key, value);
            }
        }

        Ok(values)
    }
}

struct SetVisitor<K, S, C>
where
    C: Borrow<Collector>,
{
    _marker: PhantomData<HashSet<K, S, C>>,
}

impl<K, S, C, G> Serialize for HashSetRef<'_, K, S, C, G>
where
    K: Serialize + Hash + Eq,
    C: Borrow<Collector>,
    G: Guard,
    S: BuildHasher,
{
    fn serialize<Sr>(&self, serializer: Sr) -> Result<Sr::Ok, Sr::Error>
    where
        Sr: Serializer,
    {
        serializer.collect_seq(self)
    }
}

impl<K, S, C> Serialize for HashSet<K, S, C>
where
    K: Serialize + Hash + Eq,
    S: BuildHasher,
    C: Borrow<Collector>,
{
    fn serialize<Sr>(&self, serializer: Sr) -> Result<Sr::Ok, Sr::Error>
    where
        Sr: Serializer,
    {
        self.pin().serialize(serializer)
    }
}

impl<'de, K, S> Deserialize<'de> for HashSet<K, S>
where
    K: Deserialize<'de> + Hash + Eq,
    S: Default + BuildHasher,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(SetVisitor::<K, S, Collector>::new())
    }
}

impl<'de, K, S> Deserialize<'de> for HashSet<K, S, Arc<Collector>>
where
    K: Deserialize<'de> + Hash + Eq + Send + 'static,
    S: Default + BuildHasher,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(SetVisitor::<K, S, Arc<Collector>>::new())
    }
}

impl<K, S> SetVisitor<K, S, Collector> {
    pub(crate) fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}
impl<K, S> SetVisitor<K, S, Arc<Collector>> {
    pub(crate) fn new() -> SetVisitor<K, S, Arc<Collector>> {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<'de, K, S> Visitor<'de> for SetVisitor<K, S, Collector>
where
    K: Deserialize<'de> + Hash + Eq,
    S: Default + BuildHasher,
{
    type Value = HashSet<K, S>;

    fn expecting(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "a set")
    }

    fn visit_seq<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: SeqAccess<'de>,
    {
        let values = match access.size_hint() {
            Some(size) => HashSet::with_capacity_and_hasher(size, S::default()),
            None => HashSet::default(),
        };

        {
            let values = values.pin();
            while let Some(key) = access.next_element()? {
                values.insert(key);
            }
        }

        Ok(values)
    }
}

impl<'de, K, S> Visitor<'de> for SetVisitor<K, S, Arc<Collector>>
where
    K: Deserialize<'de> + Hash + Eq + Send + 'static,
    S: Default + BuildHasher,
{
    type Value = HashSet<K, S, Arc<Collector>>;

    fn expecting(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "a set")
    }

    fn visit_seq<M>(self, mut access: M) -> Result<Self::Value, M::Error>
    where
        M: SeqAccess<'de>,
    {
        let values = match access.size_hint() {
            Some(size) => HashSet::builder()
                .hasher(S::default())
                .shared_collector(Arc::new(Collector::new()))
                .capacity(size)
                .build(),
            None => HashSet::builder()
                .hasher(S::default())
                .shared_collector(Arc::new(Collector::new()))
                .capacity(0)
                .build(),
        };

        {
            let values = values.pin();
            while let Some(key) = access.next_element()? {
                values.insert(key);
            }
        }

        Ok(values)
    }
}

#[cfg(test)]
mod test {
    use crate::HashMap;
    use crate::HashSet;

    #[test]
    fn test_map() {
        let map: HashMap<u8, u8> = HashMap::new();
        let guard = map.guard();

        map.insert(0, 4, &guard);
        map.insert(1, 3, &guard);
        map.insert(2, 2, &guard);
        map.insert(3, 1, &guard);
        map.insert(4, 0, &guard);

        let serialized = serde_json::to_string(&map).unwrap();
        let deserialized = serde_json::from_str(&serialized).unwrap();

        assert_eq!(map, deserialized);
    }

    #[test]
    fn test_set() {
        let map: HashSet<u8> = HashSet::new();
        let guard = map.guard();

        map.insert(0, &guard);
        map.insert(1, &guard);
        map.insert(2, &guard);
        map.insert(3, &guard);
        map.insert(4, &guard);

        let serialized = serde_json::to_string(&map).unwrap();
        let deserialized = serde_json::from_str(&serialized).unwrap();

        assert_eq!(map, deserialized);
    }
}
